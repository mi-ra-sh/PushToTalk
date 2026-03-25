"""
Push-to-Talk with Whisper GPU for Windows
Multi-engine: HuggingFace transformers (+ LoRA) / faster-whisper (CTranslate2)
Modular architecture: config → audio → engines → text → UI → I/O
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
    LANG_CONFIGS, HOTKEY_NAMES, MODELS, SOUNDS, BASE_DIR,
    WHISPER_MODEL_NAMES, FASTER_WHISPER_MODEL_NAMES, LORA_COMPATIBLE_MODELS,
)

logger = setup_logging()
config = load_config()

from ui import TrayManager, RecordingIndicator

# Show loading icon immediately
tray = TrayManager(config, callbacks={})
tray.show_loading()

# === Phase 2: Heavy imports (tray is already visible) ===
logger.info("Loading STT engine...")

from audio_engine import AudioRecorder, process_audio
from engines import STTEngine, TranscriptionResult, check_vram
from text_processing import post_process, is_hallucination
from input_output import (
    get_keyboard_language, get_active_window_info, detect_paste_mode,
    paste_text as io_paste_text,
)
from pynput import keyboard
from pynput.keyboard import Key, KeyCode

import sounddevice as sd


# === Engine Factory ===

def create_engine(model_id: str) -> STTEngine:
    """Create an STT engine for the given model ID."""
    if model_id in WHISPER_MODEL_NAMES:
        from engines.whisper_engine import WhisperEngine
        return WhisperEngine(
            model_id=model_id,
            model_name=WHISPER_MODEL_NAMES[model_id],
            config=config,
        )
    elif model_id in FASTER_WHISPER_MODEL_NAMES:
        from engines.faster_whisper_engine import FasterWhisperEngine
        return FasterWhisperEngine(
            model_id=model_id,
            model_name=FASTER_WHISPER_MODEL_NAMES[model_id],
        )
    else:
        raise ValueError(f"Unknown model: {model_id}")


# === App State ===

class AppState:
    """Central state for the application."""

    def __init__(self, config: dict):
        self.config = config
        self.language = config["language"]
        self.is_running = True
        self.history = load_history()
        self.engine = None
        self.engine_status = "loading"  # "loaded", "loading", "offline"
        self._switching = False

    def add_to_history(self, result: TranscriptionResult):
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
        self.language = "uk" if self.language == "en" else "en"
        self.config["language"] = self.language
        save_config(self.config)
        label = LANG_CONFIGS[self.language]["label"]
        play_sound("success")
        logger.info(f"Мова: {label}")

    def sync_language_with_keyboard(self):
        kb_lang = get_keyboard_language()
        if kb_lang != self.language:
            self.language = kb_lang
            label = LANG_CONFIGS[self.language]["label"]
            logger.info(f"Розкладка → {label}")


# === Sound playback ===

def play_sound(sound_name):
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
    os.makedirs(ORGANIC_AUDIO_DIR, exist_ok=True)
    if not os.path.exists(ORGANIC_METADATA):
        with open(ORGANIC_METADATA, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerow(["file_name", "transcription", "category", "source", "duration"])
    logger.info(f"Organic data collection: {ORGANIC_DATA_DIR}")


def get_next_organic_id():
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

# Load initial engine
selected = config["selected_model"]
try:
    state.engine = create_engine(selected)
    if not check_vram(state.engine):
        play_sound("error")
        state.engine_status = "offline"
    else:
        state.engine.load()
        state.engine_status = "loaded" if state.engine.is_loaded() else "offline"
except Exception as e:
    logger.error(f"Engine load error: {e}")
    import traceback
    traceback.print_exc()
    state.engine_status = "offline"

recorder = AudioRecorder(
    sample_rate=16000,
    device_name=config["input_device"],
    max_seconds=config["max_recording_seconds"],
)
indicator = RecordingIndicator()

if config["collect_organic_data"]:
    init_organic_collection()


# === Engine switching ===

def switch_engine(new_model_id: str):
    """Switch to a different STT engine. Runs in background thread."""
    if state._switching:
        logger.warning("Already switching engine, please wait")
        return
    if new_model_id == config["selected_model"] and state.engine_status == "loaded":
        return

    def _switch():
        state._switching = True
        state.engine_status = "loading"
        tray.show_model_loading()

        try:
            # Unload current
            if state.engine:
                state.engine.unload()
                state.engine = None

            # LoRA only for compatible models
            if new_model_id not in LORA_COMPATIBLE_MODELS:
                config["use_lora"] = False

            # Load new
            state.engine = create_engine(new_model_id)
            if not check_vram(state.engine):
                play_sound("error")
                state.engine_status = "offline"
                logger.error(f"Cannot switch to {MODELS[new_model_id]}: not enough VRAM")
                return

            state.engine.load()

            config["selected_model"] = new_model_id
            save_config(config)

            state.engine_status = "loaded" if state.engine.is_loaded() else "offline"
            logger.info(f"Switched to {MODELS[new_model_id]}")
        except Exception as e:
            logger.error(f"Engine switch error: {e}")
            import traceback
            traceback.print_exc()
            state.engine_status = "offline"
        finally:
            state._switching = False
            tray._is_loading = False
            tray.setup()

    threading.Thread(target=_switch, daemon=True).start()


# === Hotkey resolution ===

def resolve_hotkey(hotkey_name: str):
    mapping = {
        "ctrl_r": Key.ctrl_r,
        "ctrl_l": Key.ctrl_l,
        "scroll_lock": Key.scroll_lock,
    }
    if hotkey_name in mapping:
        return mapping[hotkey_name]
    if hotkey_name.startswith("f") and hotkey_name[1:].isdigit():
        fnum = int(hotkey_name[1:])
        if 13 <= fnum <= 20:
            return KeyCode.from_vk(0x7C + (fnum - 13))
    return Key.ctrl_r


current_hotkey = resolve_hotkey(config["hotkey"])


# === Tray callbacks ===

def on_toggle_language():
    state.toggle_language()
    tray.update_icon()
    tray.update_menu()


def on_change_model(model_id: str):
    switch_engine(model_id)


def on_toggle_lora():
    if config["selected_model"] not in LORA_COMPATIBLE_MODELS:
        return

    config["use_lora"] = not config["use_lora"]
    save_config(config)

    lora_state = "ON" if config["use_lora"] else "OFF"
    logger.info(f"LoRA: {lora_state}, reloading...")

    # Re-create engine with new LoRA setting
    switch_engine(config["selected_model"])


def on_refresh_model():
    logger.info("Refreshing model...")
    try:
        if state.engine:
            state.engine.unload()
        state.engine = create_engine(config["selected_model"])
        state.engine.load()
        state.engine_status = "loaded" if state.engine.is_loaded() else "offline"
        logger.info("Model refreshed")
    except Exception as e:
        logger.error(f"Refresh error: {e}")
        state.engine_status = "offline"


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
    "get_engine_status": lambda: state.engine_status,
}


# === Real-time preview ===

_preview_thread = None
_preview_stop = threading.Event()


def _realtime_preview_loop():
    interval = 2.0
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
            processed = process_audio(audio, sample_rate=16000, use_vad=False)
            if len(processed) / 16000 < 0.3:
                continue

            # Preview only available for WhisperEngine
            if hasattr(state.engine, "preview_transcribe"):
                preview_text = state.engine.preview_transcribe(processed, state.language, sample_rate=16000)
                if preview_text:
                    logger.info(f"   [preview] {preview_text[:60]}...")
        except Exception as e:
            logger.debug(f"Preview error: {e}")


def start_preview():
    global _preview_thread
    _preview_stop.clear()
    _preview_thread = threading.Thread(target=_realtime_preview_loop, daemon=True)
    _preview_thread.start()


def stop_preview():
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

    # Check engine is ready
    if state.engine is None or not state.engine.is_loaded():
        logger.warning("Engine not loaded")
        play_sound("error")
        indicator.hide()
        return

    # Transcribe
    result = state.engine.transcribe(audio, state.language)

    if result is None:
        indicator.hide()
        return

    # Hallucination check (safety net for all engines)
    if is_hallucination(result.text):
        logger.info(f"[!!] {result.text}")
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
    model_short = MODELS.get(config["selected_model"], "?")
    logger.info(
        f"[{result.language.upper()}]{conf_str} {result.text}  "
        f"({result.inference_time:.1f}s | {result.audio_duration:.1f}s | {model_short})"
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

    # Ctrl+Alt+L — toggle language
    if _ctrl_held and _alt_held and hasattr(key, "vk") and key.vk == 76:
        on_toggle_language()
        return

    # Start recording
    if key == current_hotkey and not recorder.is_recording and not _alt_held:
        if state.engine_status != "loaded":
            play_sound("error")
            logger.warning("Engine not ready")
            return

        if config["sync_lang_with_keyboard"]:
            state.sync_language_with_keyboard()
            tray.update_icon()

        recorder.start_recording()
        play_sound("start")
        logger.info("Recording...")
        indicator.update_status("#00CC00")
        indicator.show()

        if config.get("realtime_preview"):
            start_preview()


def on_release(key):
    global _ctrl_held, _alt_held

    if key in (Key.ctrl_l, Key.ctrl_r):
        _ctrl_held = False
    if key in (Key.alt_l, Key.alt_r, Key.alt_gr):
        _alt_held = False

    if key == current_hotkey and recorder.is_recording:
        if config.get("realtime_preview"):
            stop_preview()

        time.sleep(0.3)  # Tail delay for last word
        play_sound("end")
        logger.info("Processing...")
        threading.Thread(target=transcribe_and_paste, daemon=True).start()


# === Main ===

def main():
    tray.setup()

    model_name = MODELS.get(config["selected_model"], config["selected_model"])
    if state.engine and hasattr(state.engine, "model_info"):
        model_label = state.engine.model_info
    else:
        model_label = f"{model_name} [{state.engine_status}]"

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
            hf_models = [WHISPER_MODEL_NAMES[m] for m in WHISPER_MODEL_NAMES if m in MODELS]
            results = check_model_updates(hf_models)
            tray.set_model_updates(results)
            tray.update_menu()

            for model_id, info in results.items():
                name = model_id.split("/")[-1]
                if info.get("updated"):
                    logger.info(f"[!] HF update: {name} (changed {info.get('last_modified', '')[:10]})")
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
