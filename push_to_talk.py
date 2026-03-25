"""
Push-to-Talk з Whisper GPU для Windows
HuggingFace transformers + PEFT (LoRA) для inference
Modular architecture: config → audio → whisper → text → UI → I/O
"""

import sys
import os

# Fix Windows console encoding and buffering
if sys.platform == "win32":
    os.system("chcp 65001 >nul 2>&1")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class Unbuffered:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        try:
            self.stream.write(data)
            self.stream.flush()
        except Exception:
            pass

    def writelines(self, datas):
        try:
            self.stream.writelines(datas)
            self.stream.flush()
        except Exception:
            pass

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


try:
    sys.stdout = Unbuffered(sys.stdout)
    sys.stderr = Unbuffered(sys.stderr)
except Exception:
    pass

# === Phase 1: Early tray icon (before heavy imports) ===
import time
import threading
import csv
import wave

import numpy as np

from config import (
    load_config, save_config, setup_logging, load_history, save_history,
    LANG_CONFIGS, HOTKEY_NAMES, SOUNDS, BASE_DIR,
)

logger = setup_logging()
config = load_config()

from ui import TrayManager, RecordingIndicator

# Show loading icon immediately
tray = TrayManager(config, callbacks={})  # callbacks set later
tray.show_loading()

# === Phase 2: Heavy imports (tray is already visible) ===
logger.info("Loading Whisper model (HuggingFace transformers)...")

from audio_engine import AudioRecorder, process_audio
from whisper_engine import WhisperEngine, TranscriptionResult
from text_processing import post_process
from input_output import (
    get_keyboard_language, get_active_window_info, detect_paste_mode,
    paste_text as io_paste_text,
)
from pynput import keyboard
from pynput.keyboard import Key, KeyCode

import sounddevice as sd


# === App State ===

class AppState:
    """Central state for the application."""

    def __init__(self, config: dict):
        self.config = config
        self.language = config["language"]
        self.is_running = True
        self.history = load_history()
        self.transcription_count = 0

    def add_to_history(self, result: TranscriptionResult):
        """Add transcription result to history."""
        entry = {
            "text": result.text,
            "language": result.language,
            "confidence": round(result.confidence, 3),
            "duration": round(result.audio_duration, 1),
            "time": time.strftime("%H:%M:%S"),
        }
        self.history.append(entry)
        if len(self.history) > 50:
            self.history = self.history[-50:]
        save_history(self.history)

    def toggle_language(self):
        """Toggle between EN and UK."""
        self.language = "uk" if self.language == "en" else "en"
        self.config["language"] = self.language
        save_config(self.config)
        label = LANG_CONFIGS[self.language]["label"]
        play_sound("success")
        logger.info(f"Мова: {label}")

    def sync_language_with_keyboard(self):
        """Sync Whisper language with Windows keyboard layout."""
        kb_lang = get_keyboard_language()
        if kb_lang != self.language:
            self.language = kb_lang
            label = LANG_CONFIGS[self.language]["label"]
            logger.info(f"Розкладка → {label}")


# === Sound playback ===

def play_sound(sound_name):
    """Play a beep sound in background thread."""
    freq, duration = SOUNDS.get(sound_name, (440, 100))
    volume = config.get("sound_volume", 0.1)

    def _play():
        try:
            sample_rate = 44100
            t = np.linspace(0, duration / 1000, int(sample_rate * duration / 1000), False)
            tone = np.sin(2 * np.pi * freq * t) * volume
            sd.play(tone.astype(np.float32), sample_rate)
        except Exception:
            pass

    threading.Thread(target=_play, daemon=True).start()


# === Organic data collection ===

ORGANIC_DATA_DIR = os.path.join(BASE_DIR, "training_data")
ORGANIC_AUDIO_DIR = os.path.join(ORGANIC_DATA_DIR, "audio")
ORGANIC_METADATA = os.path.join(ORGANIC_DATA_DIR, "metadata.csv")


def init_organic_collection():
    """Initialize organic data collection directories."""
    os.makedirs(ORGANIC_AUDIO_DIR, exist_ok=True)
    if not os.path.exists(ORGANIC_METADATA):
        with open(ORGANIC_METADATA, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerow(["file_name", "transcription", "category", "source", "duration"])
    logger.info(f"Organic data collection: {ORGANIC_DATA_DIR}")


def get_next_organic_id():
    """Get next ID for organic recording."""
    max_id = 0
    if os.path.exists(ORGANIC_METADATA):
        with open(ORGANIC_METADATA, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                try:
                    fid = int(row.get("file_name", "").replace(".wav", ""))
                    max_id = max(max_id, fid)
                except ValueError:
                    pass
    return max_id + 1


def save_organic_data(audio_np, text, duration):
    """Save audio + transcription for organic collection."""
    try:
        file_id = get_next_organic_id()
        file_name = f"{file_id:04d}.wav"
        file_path = os.path.join(ORGANIC_AUDIO_DIR, file_name)

        peak = np.max(np.abs(audio_np))
        if peak > 0:
            audio_norm = audio_np * (0.891 / peak)
        else:
            audio_norm = audio_np
        audio_int16 = (audio_norm * 32767).astype(np.int16)

        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_int16.tobytes())

        with open(ORGANIC_METADATA, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerow([file_name, text, "ORGANIC", f"organic_{file_id}", f"{duration:.2f}"])
    except Exception as e:
        logger.warning(f"Organic save error: {e}")


# === Initialize components ===

state = AppState(config)

try:
    whisper = WhisperEngine(config)
except Exception as e:
    logger.error(f"Model load error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

recorder = AudioRecorder(
    sample_rate=16000,
    device_name=config["input_device"],
    max_seconds=config["max_recording_seconds"],
)
indicator = RecordingIndicator()

if config["collect_organic_data"]:
    init_organic_collection()


# === Hotkey resolution ===

def resolve_hotkey(hotkey_name: str):
    """Convert hotkey config string to pynput Key object."""
    mapping = {
        "ctrl_r": Key.ctrl_r,
        "ctrl_l": Key.ctrl_l,
        "scroll_lock": Key.scroll_lock,
    }
    if hotkey_name in mapping:
        return mapping[hotkey_name]
    # F13-F20
    if hotkey_name.startswith("f") and hotkey_name[1:].isdigit():
        fnum = int(hotkey_name[1:])
        if 13 <= fnum <= 20:
            return KeyCode.from_vk(0x7C + (fnum - 13))
    return Key.ctrl_r  # fallback


current_hotkey = resolve_hotkey(config["hotkey"])


# === Tray callbacks ===

def on_toggle_language():
    state.toggle_language()
    tray.update_icon()
    tray.update_menu()


def on_change_model(model_name):
    """Switch to a different Whisper model."""
    if model_name == config["model_name"]:
        return

    old_name = config["model_name"].split("/")[-1]
    new_name = model_name.split("/")[-1]
    logger.info(f"Switching model: {old_name} -> {new_name}...")

    config["model_name"] = model_name
    # LoRA only for whisper-large-v3
    if model_name != "openai/whisper-large-v3":
        config["use_lora"] = False
    save_config(config)

    tray.show_model_loading()

    def _switch():
        try:
            whisper.switch_model(model_name, config["use_lora"])
            whisper.transcription_count = 0
            tray._is_loading = False
            tray.setup()
            logger.info(f"Model switched to {new_name}")
        except Exception as e:
            logger.error(f"Model switch error: {e}")
            tray._is_loading = False
            tray.setup()

    threading.Thread(target=_switch, daemon=True).start()


def on_toggle_lora():
    """Toggle LoRA adapter on/off."""
    if config["model_name"] != "openai/whisper-large-v3":
        return

    config["use_lora"] = not config["use_lora"]
    save_config(config)

    lora_state = "ON" if config["use_lora"] else "OFF"
    logger.info(f"LoRA: {lora_state}, reloading...")

    tray.show_model_loading()

    def _reload():
        try:
            whisper.switch_model(config["model_name"], config["use_lora"])
            tray._is_loading = False
            tray.setup()
            logger.info(f"LoRA {lora_state}, model reloaded")
        except Exception as e:
            logger.error(f"LoRA toggle error: {e}")
            tray._is_loading = False
            tray.setup()

    threading.Thread(target=_reload, daemon=True).start()


def on_refresh_model():
    logger.info("Refreshing model...")
    try:
        whisper.reload_model(silent=False)
        whisper.transcription_count = 0
        logger.info("Model refreshed, counter reset")
    except Exception as e:
        logger.error(f"Refresh error: {e}")


def on_exit():
    logger.info("Exiting...")
    state.is_running = False
    tray.stop()
    os._exit(0)


def on_change_hotkey(hotkey_name: str):
    global current_hotkey
    current_hotkey = resolve_hotkey(hotkey_name)
    config["hotkey"] = hotkey_name
    save_config(config)
    display = HOTKEY_NAMES.get(hotkey_name, hotkey_name)
    logger.info(f"Hotkey changed to: {display}")
    tray.update_menu()


# Wire up tray callbacks
tray.callbacks = {
    "on_toggle_language": on_toggle_language,
    "on_change_model": on_change_model,
    "on_toggle_lora": on_toggle_lora,
    "on_refresh_model": on_refresh_model,
    "on_exit": on_exit,
    "on_change_hotkey": on_change_hotkey,
    "get_language": lambda: state.language,
    "get_history": lambda: state.history,
}


# === Real-time preview ===

_preview_thread = None
_preview_stop = threading.Event()


def _realtime_preview_loop():
    """Background thread: periodically transcribe current buffer for preview."""
    interval = 2.0  # seconds between preview attempts
    while not _preview_stop.is_set():
        _preview_stop.wait(interval)
        if _preview_stop.is_set():
            break
        if not recorder.is_recording:
            break

        audio = recorder.get_current_buffer()
        if audio is None or len(audio) / 16000 < 0.5:
            continue

        try:
            from audio_engine import process_audio
            processed = process_audio(audio, sample_rate=16000, use_vad=False)
            if len(processed) / 16000 < 0.3:
                continue

            preview_text = whisper.preview_transcribe(processed, state.language, sample_rate=16000)
            if preview_text:
                logger.info(f"   [preview] {preview_text[:60]}...")
        except Exception as e:
            logger.debug(f"Preview error: {e}")


def start_preview():
    """Start real-time preview thread."""
    global _preview_thread
    _preview_stop.clear()
    _preview_thread = threading.Thread(target=_realtime_preview_loop, daemon=True)
    _preview_thread.start()


def stop_preview():
    """Stop real-time preview thread."""
    _preview_stop.set()


# === Transcription pipeline ===

def transcribe_and_paste():
    """Main pipeline: audio → process → transcribe → post-process → paste."""
    indicator.update_status("#FFCC00")  # Yellow = processing

    audio = recorder.stop_recording()
    if audio is None or len(audio) == 0:
        logger.warning("No recording")
        play_sound("error")
        indicator.hide()
        return

    duration = len(audio) / 16000
    rms = np.sqrt(np.mean(audio**2))

    if duration < 0.3:
        logger.debug("[--] too short")
        indicator.hide()
        return

    if rms < 0.002:
        logger.debug("[~~] too quiet")
        indicator.hide()
        return

    # Save raw audio for organic collection
    raw_audio = audio.copy() if config["collect_organic_data"] else None

    # Process audio
    audio = process_audio(audio, sample_rate=16000)

    # Transcribe
    result = whisper.transcribe(audio, state.language, sample_rate=16000)

    if result is None:
        indicator.hide()
        return

    # Post-process text
    result.text = post_process(result.text, result.language, config.get("custom_vocabulary"))

    if not result.text:
        indicator.hide()
        return

    # Hide indicator before paste
    indicator.hide()

    # Detect window and paste
    window_title, process_name = get_active_window_info()
    paste_mode = detect_paste_mode(window_title, process_name)

    logger.debug(f"Window: {window_title[:40]}")

    if config["auto_paste"]:
        io_paste_text(result.text, paste_mode, config["auto_enter"])

    # Organic collection
    if config["collect_organic_data"] and raw_audio is not None:
        threading.Thread(
            target=save_organic_data,
            args=(raw_audio, result.text, duration),
            daemon=True,
        ).start()

    # Add to history
    state.add_to_history(result)
    tray.update_menu()

    # Confidence indicator
    conf_str = ""
    if result.confidence > 0:
        if result.confidence < 0.3:
            conf_str = " [?]"
        elif result.confidence > 0.8:
            conf_str = ""
        else:
            conf_str = f" [{result.confidence:.0%}]"

    play_sound("success")
    logger.info(
        f"[{result.language.upper()}]{conf_str} {result.text}  "
        f"({result.inference_time:.1f}s | {result.audio_duration:.1f}s audio)"
    )


# === Keyboard listener ===

_ctrl_held = False
_alt_held = False


def on_press(key):
    global _ctrl_held, _alt_held

    if key in (Key.ctrl_l, Key.ctrl_r):
        _ctrl_held = True
    if key in (Key.alt_l, Key.alt_r, Key.alt_gr):
        _alt_held = True

    # Ctrl+Alt+L — toggle language (vk 76 = 'L')
    if _ctrl_held and _alt_held and hasattr(key, "vk") and key.vk == 76:
        on_toggle_language()
        return

    # Start recording
    if key == current_hotkey and not recorder.is_recording and not _alt_held:
        if config["sync_lang_with_keyboard"]:
            state.sync_language_with_keyboard()
            tray.update_icon()

        recorder.start_recording()
        play_sound("start")
        logger.info("Recording...")
        indicator.update_status("#00CC00")  # Green
        indicator.show()

        # Start real-time preview if enabled
        if config.get("realtime_preview"):
            start_preview()


def on_release(key):
    global _ctrl_held, _alt_held

    if key in (Key.ctrl_l, Key.ctrl_r):
        _ctrl_held = False
    if key in (Key.alt_l, Key.alt_r, Key.alt_gr):
        _alt_held = False

    if key == current_hotkey and recorder.is_recording:
        # Stop preview if running
        if config.get("realtime_preview"):
            stop_preview()

        time.sleep(0.3)  # Tail delay for last word
        play_sound("end")
        logger.info("Processing...")
        threading.Thread(target=transcribe_and_paste, daemon=True).start()


# === Main ===

def main():
    # Setup tray with full menu
    tray.setup()

    model_label = whisper.model_info
    hotkey_display = HOTKEY_NAMES.get(config["hotkey"], config["hotkey"])

    logger.info("")
    logger.info("=" * 50)
    logger.info("  PUSH-TO-TALK READY")
    logger.info("=" * 50)
    logger.info(f"   Hotkey: {hotkey_display}")
    logger.info(f"   Model: {model_label}")
    logger.info(f"   Language: {LANG_CONFIGS[state.language]['label']} (Ctrl+Alt+L)")
    logger.info(f"   Auto-paste: {'ON' if config['auto_paste'] else 'OFF'}")
    logger.info(f"   Max recording: {config['max_recording_seconds']}s")
    logger.info(f"   Organic data: {'ON' if config['collect_organic_data'] else 'OFF'}")
    logger.info("=" * 50)
    logger.info(f"   Hold {hotkey_display} → speak → release")
    logger.info("=" * 50)
    logger.info("")

    # Check for model updates in background
    def _check_updates():
        try:
            from update_checker import check_model_updates
            from config import AVAILABLE_MODELS
            results = check_model_updates(list(AVAILABLE_MODELS.keys()))
            tray.set_model_updates(results)
            tray.update_menu()

            for model_id, info in results.items():
                name = AVAILABLE_MODELS.get(model_id, model_id.split("/")[-1])
                if info.get("updated"):
                    logger.info(f"[!] HF update: {name} (changed {info.get('last_modified', '')[:10]})")
                elif info.get("error"):
                    logger.debug(f"HF check error for {name}: {info['error']}")
        except Exception as e:
            logger.debug(f"Update check error: {e}")

    threading.Thread(target=_check_updates, daemon=True).start()

    # Start audio stream
    recorder.start_stream()

    # Run keyboard listener (blocking)
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            while state.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Exit (Ctrl+C)")
        finally:
            recorder.stop_stream()
            tray.stop()
            sys.exit(0)


if __name__ == "__main__":
    main()
