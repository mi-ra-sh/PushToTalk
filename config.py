"""
PushToTalk Configuration Module
Завантаження/збереження config.json + логування
"""

import os
import json
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import OrderedDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
LOG_PATH = os.path.join(BASE_DIR, "push_to_talk.log")
HISTORY_PATH = os.path.join(BASE_DIR, "history.json")

# Default config values
DEFAULTS = {
    "hotkey": "ctrl_r",
    "language": "en",
    "auto_paste": True,
    "auto_enter": False,
    "input_device": "Voicemeeter Out B1",
    "beam_size": 5,
    "model_name": "openai/whisper-large-v3",
    "use_lora": False,
    "collect_organic_data": False,
    "sync_lang_with_keyboard": True,
    "max_recording_seconds": 120,
    "sound_volume": 0.1,
    "no_speech_threshold": 0.6,
    "realtime_preview": False,
    "custom_vocabulary": {},
}

# Hotkey display names
HOTKEY_NAMES = {
    "ctrl_r": "Right Ctrl",
    "ctrl_l": "Left Ctrl",
    "scroll_lock": "Scroll Lock",
    "f13": "F13",
    "f14": "F14",
    "f15": "F15",
    "f16": "F16",
    "f17": "F17",
    "f18": "F18",
    "f19": "F19",
    "f20": "F20",
}

# Language configs (prompt texts for Whisper)
LANG_CONFIGS = {
    "en": {
        "prompt": "Make a commit and push to remote. Merge this branch. Deploy to server. Check the Docker container and Kubernetes cluster.",
        "label": "EN",
    },
    "uk": {
        "prompt": "Зроби commit і push на remote. Проаналізуй цей код. Відкрий файл і подивись. Закрий термінал. Запусти deploy на server. Перевір Docker container і Kubernetes cluster.",
        "label": "UK",
    },
}

# Available Whisper models for switching
AVAILABLE_MODELS = OrderedDict([
    ("openai/whisper-large-v3", "whisper-large-v3"),
    ("openai/whisper-large-v3-turbo", "whisper-large-v3-turbo"),
])

# Sound configs (freq Hz, duration ms)
SOUNDS = {
    "start": (800, 150),
    "end": (600, 150),
    "success": (1000, 100),
    "error": (300, 300),
}


def load_config() -> dict:
    """Load config from JSON file, merging with defaults for missing keys."""
    config = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            config.update(user_config)
        except (json.JSONDecodeError, IOError) as e:
            logging.getLogger("ptt").warning(f"Config load error, using defaults: {e}")
    return config


def save_config(config: dict):
    """Save config to JSON file."""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logging.getLogger("ptt").error(f"Config save error: {e}")


def setup_logging() -> logging.Logger:
    """Setup logging with console (INFO) + rotating file (DEBUG)."""
    logger = logging.getLogger("ptt")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler — INFO level, compact format
    import sys
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)

    # File handler — DEBUG level, rotating (5MB x 3 backups)
    try:
        file_handler = RotatingFileHandler(
            LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(file_handler)
    except IOError:
        pass

    return logger


def load_history() -> list:
    """Load transcription history from file."""
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return []


def save_history(history: list):
    """Save transcription history to file (keep last 50)."""
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history[-50:], f, ensure_ascii=False, indent=2)
    except IOError:
        pass
