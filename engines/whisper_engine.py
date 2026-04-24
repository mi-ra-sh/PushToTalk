"""
Whisper Engine — HuggingFace transformers backend, LoRA-only path.

Retained solely for the trained LoRA adapter (targets openai/whisper-large-v3).
Non-LoRA paths are served by faster_whisper_engine (CTranslate2, float16).
Features: prompt_ids for code-switching, LoRA adapter merge, anti-loop,
hallucination detection, confidence scoring, audio chunking, preview.
"""

import os
import time
import gc
import logging
import threading
from typing import Optional

import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from engines import STTEngine, TranscriptionResult
from config import LANG_CONFIGS
from text_processing import is_hallucination

logger = logging.getLogger("ptt")

WHISPER_VRAM_MB = 3200  # HF large-v3 + merged LoRA, fp16


class WhisperEngine(STTEngine):
    """HuggingFace Whisper large-v3 + merged LoRA adapter.

    Used only when `selected_model == "whisper-v3"`. For the non-LoRA pathway
    use FasterWhisperEngine with `whisper-v3-fast` or `whisper-turbo-fast`.
    """

    def __init__(self, model_id: str, model_name: str, config: dict):
        self.model_id = model_id
        self.model_name = model_name
        self.name = f"Whisper ({model_name.split('/')[-1]}+LoRA)"
        self.VRAM_REQUIRED_MB = WHISPER_VRAM_MB

        self.beam_size = config.get("beam_size", 5)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = None
        self.processor = None
        self.prompt_ids_map = {}
        self.transcription_count = 0
        self._gpu_lock = threading.Lock()

    def load(self) -> None:
        """Load HF Whisper base, merge LoRA adapter, prepare prompt IDs."""
        logger.info(f"Loading {self.name}...")

        self.processor = WhisperProcessor.from_pretrained(
            self.model_name, language="en", task="transcribe"
        )

        adapter_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "whisper_lora_adapter",
        )
        if not os.path.isdir(adapter_dir):
            raise FileNotFoundError(
                f"LoRA adapter not found at {adapter_dir}. "
                f"WhisperEngine is LoRA-only; use faster-whisper for the base model."
            )

        from peft import PeftModel as PeftModelClass
        base = WhisperForConditionalGeneration.from_pretrained(self.model_name, dtype=torch.float32)
        logger.info(f"Loading LoRA adapter from {adapter_dir}...")
        merged = PeftModelClass.from_pretrained(base, adapter_dir).merge_and_unload()
        self.model = merged.to(self.device, self.dtype)

        self.prompt_ids_map = {
            lang_code: self.processor.get_prompt_ids(
                lang_cfg["prompt"], return_tensors="pt"
            ).to(self.device)
            for lang_code, lang_cfg in LANG_CONFIGS.items()
        }

        self.model.eval()

        vram = torch.cuda.memory_allocated() / 1024**2 if self.device == "cuda" else 0
        logger.info(f"{self.name} loaded on {self.device.upper()} ({vram:.0f}MB VRAM)")

    def unload(self) -> None:
        """Free GPU memory."""
        with self._gpu_lock:
            if self.model is not None:
                del self.model
                self.model = None
            if self.processor is not None:
                del self.processor
                self.processor = None
            self.prompt_ids_map.clear()

            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

            logger.info(f"{self.name} unloaded")

    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def transcribe(self, audio: np.ndarray, lang: str) -> Optional[TranscriptionResult]:
        """Transcribe audio with chunking, hallucination detection, confidence."""
        sample_rate = 16000
        duration = len(audio) / sample_rate

        if duration < 0.3:
            logger.debug("Audio too short (<0.3s)")
            return None

        rms = np.sqrt(np.mean(audio**2))
        if rms < 0.002:
            logger.debug("Audio too quiet (RMS < 0.002)")
            return None

        with self._gpu_lock:
            try:
                t_start = time.time()
                CHUNK_SEC = 28
                chunk_samples = CHUNK_SEC * sample_rate

                if len(audio) <= chunk_samples:
                    text, confidence = self._transcribe_chunk(audio, lang, sample_rate)
                else:
                    n_chunks = (len(audio) + chunk_samples - 1) // chunk_samples
                    logger.info(f"   [{duration:.1f}s audio -> {n_chunks} chunks]")
                    texts = []
                    confidences = []
                    for i in range(0, len(audio), chunk_samples):
                        chunk = audio[i : i + chunk_samples]
                        chunk_text, chunk_conf = self._transcribe_chunk(chunk, lang, sample_rate)
                        if chunk_text and not is_hallucination(chunk_text):
                            texts.append(chunk_text)
                            confidences.append(chunk_conf)
                    text = " ".join(texts)
                    confidence = sum(confidences) / len(confidences) if confidences else 0.0

                t_elapsed = time.time() - t_start

            except Exception as e:
                logger.error(f"Transcription error: {e}")
                return None

        if not text:
            return None

        if is_hallucination(text):
            logger.info(f"[!!] {text}")
            return None

        self.transcription_count += 1

        return TranscriptionResult(
            text=text,
            language=lang,
            confidence=confidence,
            audio_duration=duration,
            inference_time=t_elapsed,
        )

    def preview_transcribe(self, audio: np.ndarray, language: str, sample_rate: int = 16000) -> Optional[str]:
        """Quick greedy transcription for real-time preview. Non-blocking."""
        acquired = self._gpu_lock.acquire(blocking=False)
        if not acquired:
            return None

        try:
            input_features = self.processor.feature_extractor(
                audio, sampling_rate=sample_rate, return_tensors="pt"
            ).input_features.to(self.device, dtype=self.dtype)

            attention_mask = torch.ones_like(input_features[:, :, 0], dtype=torch.long)

            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    max_new_tokens=200,
                    num_beams=1,
                    language=language,
                    task="transcribe",
                    prompt_ids=self.prompt_ids_map.get(language),
                )

            text = self.processor.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()

            del input_features, predicted_ids, attention_mask
            return text if text else None

        except Exception as e:
            logger.debug(f"Preview transcribe error: {e}")
            return None
        finally:
            self._gpu_lock.release()

    def _transcribe_chunk(self, audio: np.ndarray, language: str, sample_rate: int) -> tuple:
        """Transcribe a single chunk <=28s. Returns (text, confidence)."""
        input_features = self.processor.feature_extractor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features.to(self.device, dtype=self.dtype)

        attention_mask = torch.ones_like(input_features[:, :, 0], dtype=torch.long)

        with torch.no_grad():
            output = self.model.generate(
                input_features,
                attention_mask=attention_mask,
                max_new_tokens=385,
                num_beams=self.beam_size,
                no_repeat_ngram_size=3,
                language=language,
                task="transcribe",
                prompt_ids=self.prompt_ids_map.get(language),
                return_dict_in_generate=True,
            )

        sequences = output.sequences
        text = self.processor.tokenizer.batch_decode(sequences, skip_special_tokens=True)[0].strip()
        confidence = self._compute_confidence(output)

        del input_features, sequences, attention_mask
        return text, confidence

    def _compute_confidence(self, output) -> float:
        """Compute confidence score (0.0 to 1.0) from generation scores."""
        try:
            if hasattr(output, "scores") and output.scores:
                import torch.nn.functional as F
                log_probs = []
                for i, score in enumerate(output.scores):
                    probs = F.softmax(score, dim=-1)
                    token_id = output.sequences[0, i + 1] if i + 1 < output.sequences.shape[1] else None
                    if token_id is not None:
                        token_prob = probs[0, token_id].item()
                        log_probs.append(token_prob)
                if log_probs:
                    return sum(log_probs) / len(log_probs)

            n_tokens = output.sequences.shape[1]
            if n_tokens > 5:
                return min(0.7, n_tokens / 50.0)
        except Exception as e:
            logger.debug(f"Confidence computation error: {e}")
        return 0.0

    @property
    def model_info(self) -> str:
        """Short model description for UI."""
        label = self.model_name.split("/")[-1] + " +LoRA"
        if self.device == "cuda":
            label += " (GPU fp16)"
        return label
