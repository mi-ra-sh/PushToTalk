"""
Faster-Whisper Engine — CTranslate2 backend
Supports Whisper Large V3 and V3 Turbo with float16.
2-4x faster than HuggingFace transformers, lower VRAM usage.
"""

import time
import gc
import logging
import threading
from typing import Optional

import numpy as np

from engines import STTEngine, TranscriptionResult
from config import LANG_CONFIGS

logger = logging.getLogger("ptt")

FASTER_VRAM_MB = {
    "whisper-v3-fast": 3200,
    "whisper-turbo-fast": 1800,
}


class FasterWhisperEngine(STTEngine):
    """Whisper V3 / V3 Turbo backend using faster-whisper (CTranslate2)."""

    def __init__(self, model_id: str, model_name: str):
        self.model_id = model_id
        self.model_name = model_name
        self.name = f"Fast ({model_name})"
        self.VRAM_REQUIRED_MB = FASTER_VRAM_MB.get(model_id, 3200)

        self.model = None
        self._gpu_lock = threading.Lock()

    def load(self) -> None:
        """Load model onto GPU with float16."""
        from faster_whisper import WhisperModel

        logger.info(f"Loading {self.name} ({self.model_name}, float16)...")

        self.model = WhisperModel(
            self.model_name,
            device="cuda",
            compute_type="float16",
        )

        logger.info(f"{self.name} loaded (faster-whisper float16)")

    def unload(self) -> None:
        """Free GPU memory."""
        with self._gpu_lock:
            if self.model is not None:
                del self.model
                self.model = None

            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info(f"{self.name} unloaded")

    def is_loaded(self) -> bool:
        return self.model is not None

    def transcribe(self, audio: np.ndarray, lang: str) -> Optional[TranscriptionResult]:
        """Transcribe audio using faster-whisper with VAD filter."""
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

                segments, info = self.model.transcribe(
                    audio,
                    language=lang,
                    beam_size=5,
                    vad_filter=True,
                    initial_prompt=self._get_initial_prompt(lang),
                )

                text = " ".join(seg.text.strip() for seg in segments)
                t_elapsed = time.time() - t_start

            except Exception as e:
                logger.error(f"Transcription error: {e}")
                return None

        if not text or not text.strip():
            return None

        return TranscriptionResult(
            text=text.strip(),
            language=lang,
            confidence=0.0,
            audio_duration=duration,
            inference_time=t_elapsed,
        )

    @staticmethod
    def _get_initial_prompt(lang: str) -> str:
        """Code-switching prompt from LANG_CONFIGS."""
        lang_cfg = LANG_CONFIGS.get(lang)
        if lang_cfg:
            return lang_cfg["prompt"]
        return ""
