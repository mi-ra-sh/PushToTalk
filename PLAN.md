# PushToTalk Improvement Plan

## Overview
Рефакторинг та покращення PushToTalk — розбивка монолітного `push_to_talk.py` (1271 рядків) на модулі з додаванням нових фіч.

## Фази реалізації

---

### Фаза 1: Config + Logging (фундамент для всього іншого)

**1.1 Config файл `config.json`**
Створити `config.py` — модуль для роботи з конфігом:
- Завантаження `config.json` з `C:\PushToTalk\config.json`
- Дефолтні значення для всіх параметрів
- Функція `save_config()` для збереження змін (мова, хоткей і т.д.)
- Функція `load_config()` з валідацією

Параметри з config.json:
```json
{
  "hotkey": "ctrl_r",
  "language": "en",
  "auto_paste": true,
  "auto_enter": false,
  "input_device": "Voicemeeter Out B1",
  "beam_size": 5,
  "model_name": "openai/whisper-large-v3",
  "use_lora": false,
  "collect_organic_data": false,
  "sync_lang_with_keyboard": true,
  "max_recording_seconds": 120,
  "sound_volume": 0.1,
  "custom_vocabulary": {}
}
```

**1.2 Logging модуль**
Замінити всі `print()` та `except: pass` на proper logging:
- `logging` модуль з Python stdlib
- Два хендлери: console (INFO) + файл `push_to_talk.log` (DEBUG) з RotatingFileHandler (5MB, 3 backups)
- Формат: `[HH:MM:SS] LEVEL message`
- Замінити `except: pass` → `except Exception as e: logger.debug(f"...: {e}")`

**1.3 Max recording length**
- Додати ліміт `max_recording_seconds` (default: 120с = 2 хв)
- В `audio_callback`: перевіряти загальну кількість семплів, зупиняти запис при перевищенні
- Логувати попередження

---

### Фаза 2: Модульна архітектура

Розбити `push_to_talk.py` на модулі. Нова структура:

```
C:\PushToTalk\
├── push_to_talk.py          ← main() entry point, ~80 рядків
├── config.py                ← конфіг + логування
├── audio_engine.py          ← запис, обробка аудіо
├── whisper_engine.py        ← модель, інференс, галюцинації
├── text_processing.py       ← auto-replace, пост-обробка
├── ui.py                    ← трей + індикатор
├── input_output.py          ← SendInput, paste, keyboard listener
├── config.json              ← налаштування користувача
└── ... (існуючі файли)
```

**2.1 `config.py`** (~60 рядків)
- `load_config()`, `save_config()`, `get_logger()`
- Всі дефолтні значення
- Валідація config.json

**2.2 `audio_engine.py`** (~150 рядків)
- `AudioRecorder` клас:
  - `start_recording()`, `stop_recording()` → повертає np.array
  - `audio_callback()` — sounddevice callback
  - Max recording length enforcement
  - Device discovery
- `trim_trailing_silence()`
- `compress_internal_silence()`
- `normalize_audio()`

**2.3 `whisper_engine.py`** (~200 рядків)
- `WhisperEngine` клас:
  - `__init__()` — завантажує модель
  - `transcribe(audio)` → `TranscriptionResult(text, confidence, language, duration)`
  - `reload_model()` — перезавантаження для скидання стану
  - `cleanup()` — VRAM cleanup
  - Лічильник транскрипцій + auto-reload кожні 25
- `is_hallucination()`, `is_prompt_leak()`, `has_repeated_phrase()`
- `HALLUCINATION_PATTERNS`
- Chunking logic (>28s)

**2.4 `text_processing.py`** (~50 рядків)
- `post_process(text, language)` — auto-replace + custom vocabulary
- `AUTO_REPLACE` словник
- Завантаження custom vocabulary з config

**2.5 `ui.py`** (~250 рядків)
- `TrayManager` клас:
  - `create_icon()`, `update_icon()`, `setup_menu()`
  - `show_history()` — показати історію транскрипцій
- `RecordingIndicator` клас (вже є, переносимо)
- Callbacks для меню

**2.6 `input_output.py`** (~150 рядків)
- SendInput structs + helpers (`press_key`, `release_key`, `type_unicode_char`, `type_text`)
- `paste_text(text)` — визначає тип вікна і вставляє відповідно
- `get_keyboard_language()`
- `GUITHREADINFO` struct

**2.7 `push_to_talk.py`** (main, ~80 рядків)
- `AppState` dataclass:
  - `is_recording`, `recording`, `language`, `transcription_count`
  - `history: list[TranscriptionResult]` (останні 20)
- `main()` — ініціалізація всіх компонентів
- `on_press()`, `on_release()` — keyboard callbacks
- Wiring: audio → whisper → text_processing → paste

---

### Фаза 3: Нові фічі

**3.1 VAD (Voice Activity Detection)**
- Встановити `silero-vad` (легший за webrtcvad, працює на PyTorch який вже є)
- Використати замість простого RMS фільтру в `audio_engine.py`
- Trimming початку та кінця аудіо по VAD замість RMS threshold
- Fallback на RMS якщо VAD недоступний

**3.2 Confidence scoring**
- `TranscriptionResult` вже матиме `confidence` поле
- Whisper `model.generate()` з `return_dict_in_generate=True, output_scores=True`
- Обчислити середній log-probability по токенах
- Порогове значення: < 0.3 → показати попередження "[?]" перед текстом
- Логувати confidence для всіх транскрипцій

**3.3 Custom vocabulary**
- В `config.json`: `"custom_vocabulary": {"kubernetes": "Kubernetes", "kube": "kubectl"}`
- Застосовується в `text_processing.py` після auto-replace
- Case-insensitive match → правильний регістр

**3.4 Transcription history (трей)**
- Зберігати останні 20 транскрипцій в `AppState.history`
- В меню трею → підменю "Історія" з останніми 10 записами (обрізані до 50 символів)
- Клік на запис → копіювати в clipboard
- Персистити в `history.json` (перезавантажується при старті)

**3.5 Customizable hotkey**
- В `config.json`: `"hotkey": "ctrl_r"` (можливі: ctrl_r, ctrl_l, f13-f24, scroll_lock тощо)
- В меню трею: "Клавіша: Right Ctrl" → субменю зі списком варіантів
- Зміна без перезапуску (перезапуск keyboard listener)

**3.6 Real-time partial transcription** (найскладніше)
- Whisper не підтримує streaming нативно
- Підхід: кожні ~2с під час запису → inference на поточному буфері → показати часткові результати
- Показувати в popup вікні (нове Tk window) або в tooltip трею
- При відпусканні клавіші → фінальна транскрипція замінює часткову
- Потрібен окремий thread для часткових інференсів
- **ВАЖЛИВО**: потребує значних ресурсів GPU, може сповільнити фінальну транскрипцію
- Реалізувати як opt-in фічу: `"realtime_preview": false` в config

---

## Порядок реалізації

| # | Завдання | Залежить від | Складність |
|---|----------|-------------|------------|
| 1 | config.py + config.json | — | Низька |
| 2 | Logging | config.py | Низька |
| 3 | Max recording length | — | Низька |
| 4 | audio_engine.py | config, logging | Середня |
| 5 | whisper_engine.py | config, logging | Середня |
| 6 | text_processing.py | config | Низька |
| 7 | input_output.py | config, logging | Середня |
| 8 | ui.py | config, logging | Середня |
| 9 | push_to_talk.py (AppState + wiring) | 4-8 | Середня |
| 10 | Custom vocabulary | text_processing | Низька |
| 11 | Transcription history | ui, AppState | Низька |
| 12 | Customizable hotkey | config, input_output | Середня |
| 13 | VAD (silero-vad) | audio_engine | Середня |
| 14 | Confidence scoring | whisper_engine | Середня |
| 15 | Real-time preview | audio, whisper, ui | Висока |

## Ризики та обмеження

- **Real-time preview** потребує ~2x GPU. На RTX 5070 Ti повинно працювати, але може збільшити latency фінального результату.
- **silero-vad** додає ще одну PyTorch модель в пам'ять (~2MB, мінімально).
- **Рефакторинг** може зламати VBS launcher якщо змінити точку входу — потрібно зберегти `push_to_talk.py` як entry point.
- **Config migration** — при оновленні config.json потрібен merge з дефолтами.
- **Confidence scoring** з `output_scores=True` може трохи сповільнити інференс.

## Що НЕ змінюється
- Voicemeeter preset скрипти (окремі файли, не залежать від main)
- Training pipeline (collect_training_data.py, finetune_whisper.py, evaluate_model.py)
- Launcher скрипти (.bat, .vbs) — лише переконатись що push_to_talk.py залишається entry point
