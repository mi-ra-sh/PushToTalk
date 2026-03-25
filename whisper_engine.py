"""
PushToTalk Whisper Engine
Model loading, inference, hallucination detection, confidence scoring
Model stays on GPU in float16 for fast inference and minimal RAM usage
"""

import time
import gc
import logging
import threading
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from config import LANG_CONFIGS

logger = logging.getLogger("ptt")

# Typical Whisper hallucination patterns
HALLUCINATION_PATTERNS = [
    "дякую за перегляд",
    "підписуйтесь",
    "subscribe",
    "thanks for watching",
    "thank you for watching",
    "like and subscribe",
    "не забудьте підписатися",
    "ставте лайки",
    "дякую за увагу",
    "продовження наступне",
    "субтитри зроблені",
    "subtitles by",
    "translated by",
    "www.",
    "http",
]


@dataclass
class TranscriptionResult:
    """Result of a transcription."""
    text: str
    language: str
    confidence: float
    audio_duration: float
    inference_time: float


def is_prompt_leak(text):
    """Check if text is a prompt leak (Whisper hallucinates prompt phrases on silence)."""
    text_lower = text.lower().strip()

    prompt_markers = [
        "deploy to server",
        "docker container",
        "kubernetes cluster",
        "commit і push",
        "commit and push",
        "make a commit",
        "merge this branch",
        "merge цей branch",
        "check the docker",
        "проаналізуй цей код",
        "відкрий файл",
        "закрий термінал",
        "запусти deploy",
        "перевір docker",
        "push на remote",
        "push to remote",
    ]

    matches = sum(1 for marker in prompt_markers if marker in text_lower)
    if matches >= 2:
        return True

    for lang_cfg in LANG_CONFIGS.values():
        if text_lower in lang_cfg["prompt"].lower():
            return True

    return False


def has_repeated_phrase(text, min_phrase_len=2, max_phrase_len=6, max_repeats=2):
    """Check if text contains a repeated phrase of any length.

    Looks for phrases of min..max words repeated max_repeats+ times consecutively.
    Returns the first found phrase or None.
    """
    words = text.lower().split()
    for phrase_len in range(min_phrase_len, min(max_phrase_len + 1, len(words) // max_repeats + 1)):
        for start in range(len(words) - phrase_len * max_repeats + 1):
            phrase = tuple(words[start : start + phrase_len])
            count = 1
            pos = start + phrase_len
            while pos + phrase_len <= len(words):
                if tuple(words[pos : pos + phrase_len]) == phrase:
                    count += 1
                    pos += phrase_len
                else:
                    break
            if count >= max_repeats:
                return " ".join(phrase)
    return None


def is_hallucination(text, segments=None, audio_rms=None):
    """Check if text is a Whisper hallucination."""
    lower = text.lower().strip()

    for pattern in HALLUCINATION_PATTERNS:
        if pattern in lower:
            return True

    words = lower.split()
    if len(words) >= 3:
        for i in range(len(words) - 2):
            if words[i] == words[i + 1] == words[i + 2]:
                return True

    if len(words) >= 2 and len(set(words)) == 1:
        return True

    repeated = has_repeated_phrase(lower)
    if repeated:
        logger.info(f'   [!!] Repeated phrase: "{repeated}"')
        return True

    if segments:
        no_speech_probs = [s.get("no_speech_prob", 0) for s in segments]
        if no_speech_probs and all(p > 0.8 for p in no_speech_probs):
            return True

    return False


class WhisperEngine:
    """Manages Whisper model lifecycle and transcription.

    Model stays on GPU in float16 permanently for fast inference and minimal RAM.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config["model_name"]
        self.beam_size = config["beam_size"]
        self.use_lora = config["use_lora"]

        # Device and dtype: float16 on CUDA, float32 on CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = None
        self.processor = None
        self.prompt_ids_map = {}
        self.transcription_count = 0

        self._gpu_lock = threading.Lock()

        self._load_model()

    def _load_model(self, silent=False):
        """Load or reload Whisper model + optional LoRA + prompt IDs."""
        if not silent:
            logger.info(f"Loading model '{self.model_name}'...")

        self.processor = WhisperProcessor.from_pretrained(
            self.model_name, language="en", task="transcribe"
        )

        if self.use_lora:
            # Load in float32 for LoRA merge, then convert
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name, dtype=torch.float32
            )
            import os
            adapter_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whisper_lora_adapter")
            from peft import PeftModel as PeftModelClass
            if not silent:
                logger.info(f"Loading LoRA adapter from {adapter_dir}...")
            self.model = PeftModelClass.from_pretrained(self.model, adapter_dir)
            self.model = self.model.merge_and_unload()
            self.model.to(self.device, self.dtype)
            if not silent:
                logger.info("LoRA adapter loaded and merged")
        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name, dtype=self.dtype
            ).to(self.device)

        self.prompt_ids_map = {}
        for lang_code, lang_cfg in LANG_CONFIGS.items():
            self.prompt_ids_map[lang_code] = self.processor.get_prompt_ids(
                lang_cfg["prompt"], return_tensors="pt"
            ).to(self.device)

        self.model.eval()

        if not silent:
            model_label = "HF+LoRA" if self.use_lora else "HF base"
            vram = torch.cuda.memory_allocated() / 1024**2 if self.device == "cuda" else 0
            logger.info(f"Model '{self.model_name}' loaded on {self.device.upper()} ({model_label}, {vram:.0f}MB VRAM)")

    def switch_model(self, model_name, use_lora=False):
        """Switch to a different model. Thread-safe."""
        self.model_name = model_name
        self.use_lora = use_lora
        self.reload_model(silent=False)

    def reload_model(self, silent=True):
        """Reload model to reset internal state. Thread-safe."""
        with self._gpu_lock:
            # Explicitly delete old model
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

            self._load_model(silent=silent)

            if not silent:
                logger.info(f"Reload #{self.transcription_count}")

    def transcribe(self, audio: np.ndarray, language: str, sample_rate: int = 16000) -> Optional[TranscriptionResult]:
        """Transcribe audio and return TranscriptionResult, or None on failure.

        Handles chunking for audio >28s, hallucination detection, confidence scoring.
        """
        duration = len(audio) / sample_rate

        # Pre-filters
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
                    text, confidence = self._transcribe_chunk(audio, language, sample_rate)
                else:
                    # Long audio — split into chunks
                    n_chunks = (len(audio) + chunk_samples - 1) // chunk_samples
                    logger.info(f"   [{duration:.1f}s audio -> {n_chunks} chunks]")
                    texts = []
                    confidences = []
                    for i in range(0, len(audio), chunk_samples):
                        chunk = audio[i : i + chunk_samples]
                        chunk_text, chunk_conf = self._transcribe_chunk(chunk, language, sample_rate)
                        if chunk_text and not is_hallucination(chunk_text, audio_rms=rms):
                            texts.append(chunk_text)
                            confidences.append(chunk_conf)
                    text = " ".join(texts)
                    confidence = sum(confidences) / len(confidences) if confidences else 0.0

                t_elapsed = time.time() - t_start

            except Exception as e:
                logger.error(f"Transcription error: {e}")
                return None

        # Hallucination check (works with strings, no GPU needed)
        if not text:
            return None

        if is_hallucination(text, audio_rms=rms):
            logger.info(f"[!!] {text}")
            return None

        self.transcription_count += 1

        return TranscriptionResult(
            text=text,
            language=language,
            confidence=confidence,
            audio_duration=duration,
            inference_time=t_elapsed,
        )

    def preview_transcribe(self, audio: np.ndarray, language: str, sample_rate: int = 16000) -> Optional[str]:
        """Quick greedy transcription for real-time preview.

        Non-blocking: returns None immediately if GPU is busy.
        """
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
        """Transcribe a single chunk. Returns (text, confidence). Assumes model is on GPU."""
        input_features = self.processor.feature_extractor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features.to(self.device, dtype=self.dtype)

        # Create attention mask to suppress warning
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

        # Confidence: use generation scores if available, otherwise estimate from no_speech
        confidence = self._compute_confidence(output, input_features)

        del input_features, sequences, attention_mask
        return text, confidence

    def _compute_confidence(self, output, input_features=None) -> float:
        """Compute confidence score (0.0 to 1.0).

        Uses generation scores if available, otherwise uses a heuristic based on
        output length relative to input length.
        """
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

            # Fallback: longer outputs relative to input generally mean higher confidence
            n_tokens = output.sequences.shape[1]
            if n_tokens > 5:
                return min(0.7, n_tokens / 50.0)
        except Exception as e:
            logger.debug(f"Confidence computation error: {e}")
        return 0.0

    def cleanup(self):
        """Free GPU memory. Thread-safe."""
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    @property
    def model_info(self) -> str:
        """Short model description for UI."""
        label = "whisper-large-v3"
        if self.use_lora:
            label += " +LoRA"
        if self.device == "cuda":
            label += " (GPU fp16)"
        return label
