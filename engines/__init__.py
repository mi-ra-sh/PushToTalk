"""
STT Engine Interface
All backends implement this interface for model loading/unloading and transcription.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import logging

import numpy as np

logger = logging.getLogger("ptt")


@dataclass
class TranscriptionResult:
    """Result of a transcription."""
    text: str
    language: str
    confidence: float
    audio_duration: float
    inference_time: float


class STTEngine(ABC):
    """Base class for speech-to-text engines."""

    name: str = ""
    VRAM_REQUIRED_MB: int = 0

    @abstractmethod
    def load(self) -> None:
        """Load model into GPU."""

    @abstractmethod
    def unload(self) -> None:
        """Free GPU memory."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is ready for transcription."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, lang: str) -> Optional[TranscriptionResult]:
        """Transcribe audio. Returns TranscriptionResult or None on failure."""


def check_vram(engine: STTEngine) -> bool:
    """Check if there's enough free VRAM to load the engine."""
    required = engine.VRAM_REQUIRED_MB
    if required <= 0:
        return True

    try:
        import torch
        if not torch.cuda.is_available():
            return True
        free, total = torch.cuda.mem_get_info()
        free_mb = free / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        logger.debug(f"VRAM check: need {required} MB, free {free_mb:.0f}/{total_mb:.0f} MB")
        if free_mb < required:
            logger.error(
                f"Not enough VRAM for {engine.name}: need {required} MB, "
                f"only {free_mb:.0f} MB free (total {total_mb:.0f} MB)"
            )
            return False
        return True
    except Exception as e:
        logger.warning(f"VRAM check failed: {e}")
        return True


def get_vram_info() -> tuple:
    """Return (used_mb, total_mb) of GPU VRAM. Returns (0, 0) if unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return (0.0, 0.0)
        free, total = torch.cuda.mem_get_info()
        total_mb = total / (1024 * 1024)
        used_mb = (total - free) / (1024 * 1024)
        return (used_mb, total_mb)
    except Exception:
        return (0.0, 0.0)
