# Push-to-Talk Session Context

## What was created
- **C:\PushToTalk\push_to_talk.py** - Main Push-to-Talk application (HuggingFace transformers + PEFT)
- **C:\PushToTalk\collect_training_data.py** - GUI for recording training data (500 sentences)
- **C:\PushToTalk\finetune_whisper.py** - LoRA fine-tuning script for Whisper large-v3
- **C:\PushToTalk\evaluate_model.py** - Evaluation: base vs fine-tuned WER comparison
- **C:\PushToTalk\training_sentences.txt** - 500 sentences in 5 categories for recording
- **C:\PushToTalk\training_data/** - Directory: audio/*.wav + metadata.csv
- **C:\PushToTalk\whisper_lora_adapter/** - Saved LoRA adapter (~60MB)
- **C:\PushToTalk\push_to_talk.bat** - Manual launcher with console
- **C:\PushToTalk\push_to_talk_silent.vbs** - Silent launcher (uses python.exe with hidden window)
- **C:\PushToTalk\install_task.bat** - Creates scheduled task (REMOVED — was causing dual instances)
- Auto-start configured via Startup folder VBS (sole auto-start method)
- **C:\PushToTalk\voicemeeter_preset.py** - Whisper preset (B1 only)
- **C:\PushToTalk\voicemeeter_preset_gaming.py** - Gaming voice chat preset (B1+B2)
- **C:\PushToTalk\voicemeeter_preset_music.py** - Music/singing preset (B1+B3, reverb)
- **C:\PushToTalk\preset_whisper.bat** / **preset_gaming.bat** / **preset_music.bat** - Batch launchers
- **C:\PushToTalk\create_shortcuts.ps1** - Creates desktop shortcuts with global hotkeys

## Features
- **Hotkey**: Right Ctrl (hold to record, release to transcribe)
- **Model**: HuggingFace transformers `openai/whisper-large-v3` + optional LoRA adapter
- **Language**: Ukrainian (forced) - English words recognized in context
- **Auto-paste**: Automatically pastes transcribed text
- **DCV Client support**: Character-by-character Unicode input for Amazon DCV
- **Visual indicator**: Circle with microphone icon near cursor during recording
- **System tray**: Icon with right-click menu to exit
- **Sound feedback**: Beeps for start/stop/success/error
- **Recording tail delay**: 0.3s after key release to capture last word
- **Auto-replace**: 80+ replacements (Russian→Ukrainian, tech terms, common errors, Voicemeeter phonetics)
- **Anti-hallucination**: Silence/RMS filtering, hallucination pattern detection
- **Audio normalization**: Normalize to -1dB headroom before Whisper
- **LoRA fine-tuning**: Automatic adapter loading if whisper_lora_adapter/ exists
- **Organic data collection**: Optional mode to save audio+transcription for future fine-tuning

## Architecture change: openai-whisper → HuggingFace transformers + PEFT

### Why
- openai-whisper doesn't support LoRA adapters
- Same model (whisper-large-v3), different API
- Enables fine-tuning on user's voice for better accuracy

### How inference works now
```python
# Old (openai-whisper):
import whisper
model = whisper.load_model("large-v3", device="cuda")
result = model.transcribe(audio, language="uk", beam_size=5, ...)

# New (HuggingFace transformers + PEFT):
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to("cuda")
model = PeftModel.from_pretrained(model, "whisper_lora_adapter")  # if exists
model = model.merge_and_unload()
input_features = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
predicted_ids = model.generate(input_features, language="uk", num_beams=5)
text = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
```

### Whisper parameters
```python
MODEL_NAME = "openai/whisper-large-v3"
LANGUAGE = "uk"
BEAM_SIZE = 5
# forced_decoder_ids set via processor.get_decoder_prompt_ids()
```

### Fine-tuning pipeline
1. **collect_training_data.py** — GUI: read 500 sentences, record WAV 16kHz
2. **finetune_whisper.py** — LoRA fine-tuning (rank=32, alpha=64, 8-bit quantization)
3. **evaluate_model.py** — Compare base vs fine-tuned WER
4. LoRA adapter auto-detected by push_to_talk.py on startup

### LoRA configuration
- Rank: 32, Alpha: 64, Dropout: 0.05
- Targets: q_proj, v_proj, k_proj, out_proj, fc1, fc2
- Adapter size: ~60MB (vs 3GB base model)
- Training: 8-bit quantization, gradient checkpointing, fp16

### Training data sources
- **Prepared sentences**: 500 sentences in 5 categories (Щ, MIX, UA, IT, NUM)
- **Common Voice (UK)**: Up to 5-10 hours from Mozilla Common Voice
- **Organic collection**: Optional auto-save during daily PTT use

### Pre-transcription filtering
- **Duration check**: < 0.3s → skip (`[--]` in console)
- **RMS check**: < 0.01 → skip (`[~~]` in console)
- **Audio normalization**: normalize to -1dB headroom before Whisper

### Post-transcription filtering
- **Hallucination patterns**: YouTube-style phrases, repeated words 3+, single-word repetitions
- Console output: `[!!]` for rejected hallucinations

### Auto-replace dictionary
Contains 80+ replacements for:
- Russian→Ukrainian words (что→що, это→це, привіт→привіт, добре, зараз, потрібно, etc.)
- Git/DevOps terms (commit, merge, branch, push, deploy, pipeline, etc.)
- Common transcription errors (сроби→зроби, нада→треба, etc.)

## Audio pipeline
- **Microphone**: PD100X → Voicemeeter Potato Hardware Input 1
- **Processing**: Gate + Compressor + EQ + Denoiser (all in Voicemeeter Potato)
- **Output**: Bus B1 → Push-to-Talk app (INPUT_DEVICE = "Voicemeeter Out B1")
- **Equalizer APO**: Completely removed (files + registry cleaned)

## Key technical details
- Python 3.11 (Python 3.14 doesn't support PyTorch CUDA)
- **HuggingFace transformers + PEFT** (was: openai-whisper) — uses ~10GB VRAM with large-v3
- SendInput with KEYEVENTF_UNICODE for character input
- pystray for system tray
- tkinter for visual indicator
- Audio stream: blocksize=4096, dtype=float32, 16kHz mono

## Dependencies
```
pip install transformers peft datasets[audio] accelerate bitsandbytes evaluate jiwer soundfile tensorboard sounddevice numpy pyperclip pynput pywin32 pillow pystray torch
```

## Known issues & solutions
- **Two instances on boot**: Had both scheduled task AND startup folder VBS — scheduled task deleted, only VBS remains
- **Russian words in output**: Fixed by forcing LANGUAGE="uk" + auto-replace dictionary
- **Whisper hallucinations**: Fixed by RMS/duration pre-filtering + hallucination pattern detection
- "No recording" error: Restart the app (audio stream may freeze)
- DCV mixed language: Works via character-by-character Unicode input
- Python 3.14: Not supported, use Python 3.11
- **bitsandbytes on Windows**: May need special Windows build (`pip install bitsandbytes-windows`)

## Changelog

### 2026-02-06 (Fine-tuning update)
- **Changed**: Migrated from openai-whisper to HuggingFace transformers + PEFT
- **Added**: LoRA adapter support — auto-loads from whisper_lora_adapter/ if present
- **Added**: collect_training_data.py — GUI for recording 500 training sentences
- **Added**: finetune_whisper.py — LoRA fine-tuning with 8-bit quantization
- **Added**: evaluate_model.py — WER evaluation with category breakdown
- **Added**: training_sentences.txt — 500 sentences in 5 categories
- **Added**: Organic data collection mode (COLLECT_ORGANIC_DATA toggle)
- **Removed**: openai-whisper dependency (replaced by transformers)
- **Removed**: INITIAL_PROMPT, TEMPERATURE, COMPRESSION_RATIO_THRESHOLD (not needed with HF API)

### 2026-02-06 (earlier)
- **Fixed**: ALLOWED_LANGUAGES undefined variable bug (line 588)
- **Added**: Pre-transcription silence filtering (duration < 0.3s, RMS < 0.01)
- **Added**: Audio normalization to -1dB headroom
- **Added**: Post-transcription hallucination detection (patterns, repeated words, no_speech_prob)
- **Added**: Transcription time logging
- **Added**: Audio callback status error logging
- **Changed**: Audio blocksize 1024 → 4096, explicit dtype='float32'
- **Added**: 80+ AUTO_REPLACE entries (was empty `{}`)
- **Updated**: SESSION_CONTEXT.md to reflect openai-whisper (not faster-whisper)
- **Added**: Recording tail delay (0.3s after key release) to prevent last-word cutoff
- **Added**: Voicemeeter phonetic variants in AUTO_REPLACE ("8-метр потайту"→"Voicemeeter Potato", etc.)
- **Fixed**: Dual-instance on boot — deleted scheduled task, kept Startup folder VBS only
- **Removed**: Equalizer APO completely (files + registry)
- **Created**: Three Voicemeeter presets: Whisper, Gaming, Music (Python scripts + .bat + desktop shortcuts)
- **Created**: Desktop shortcuts with global hotkeys (Ctrl+Alt+1/2/3) via create_shortcuts.ps1
- **Changed**: A1 monitoring OFF in all presets (was causing echo)

### 2026-02-02
- **Fixed**: Launcher not working (pythonw.exe silently crashed)
- **Fixed**: Russian words appearing in transcription

## Voicemeeter Potato setup for PD100X

### Gate (on channel strip)
- Threshold: ~-40dB
- Attack: ~1ms
- Release: ~100ms

### Compressor (on channel strip)
- Ratio: 3-4:1
- Threshold: ~-20dB
- Makeup gain: +3-5dB

### EQ (click EQ on Strip 1 — 6 bands available)
- Band 1: 80Hz HPF — remove low-end rumble
- Band 2: 200Hz -2dB Q=1.0 — reduce boominess
- Band 3: 1kHz +2dB Q=1.0 — clarity
- Band 4: 3kHz +3dB Q=1.5 — presence/intelligibility
- Band 5: 6kHz +2dB Q=1.5 — brightness
- Band 6: 14kHz -2dB Q=1.0 — reduce sibilance

Note: Voicemeeter Potato strip EQ has 6 bands (not 8). Must be set manually in GUI.

### Denoiser
- Enable built-in denoiser on mic channel

### Routing
- PD100X → Hardware Input 1 → processing (Gate+Comp+EQ+Denoiser) → Bus B1 → Push-to-Talk
- A1 OFF on Strip 1 (monitoring disabled in all presets)
- Equalizer APO: completely removed (files + registry)

### Voicemeeter Presets (desktop shortcuts with global hotkeys)
| Preset | Shortcut | Bus routing | Comp | Gate | Denoiser | Reverb | Use case |
|--------|----------|-------------|------|------|----------|--------|----------|
| Whisper | Ctrl+Alt+1 | B1 | 1 | 1 | 4 | 0 | Push-to-Talk / Whisper STT |
| Gaming | Ctrl+Alt+2 | B1+B2 | 4 | 2 | 6 | 0 | Discord / TeamSpeak voice chat |
| Music | Ctrl+Alt+3 | B1+B3 | 5 | 1 | 2 | 4 | Singing & guitar recording/streaming |

All presets: Gain=+6dB, Mono=ON, A1=OFF, EQ=ON (manual bands)

## Voicemeeter Remote API — уроки

DLL: `C:\Program Files (x86)\VB\Voicemeeter\VoicemeeterRemote64.dll`

### Що працює
- **`VBVMR_SetParameters()`** (macro format) — ПРАЦЮЄ для всього крім Strip EQ
- Формат: `b"Strip[0].Gain=6;Strip[0].Mono=1;Strip[0].Comp=1;"` (крапка з комою між параметрами)
- **`VBVMR_SetParameterFloat()`** — повертає 0 (OK) навіть коли НЕ записує! Не довіряти return code
- **`VBVMR_GetParameterFloat()`** — читає правильно, використовувати для верифікації

### Параметри що працюють через API
```
Strip[0].A1, .A2, .A3, .A4, .A5     — routing to hardware buses
Strip[0].B1, .B2, .B3               — routing to virtual buses
Strip[0].Gain                        — fader gain (dB)
Strip[0].Mono                        — mono on/off
Strip[0].Mute                        — mute on/off
Strip[0].Comp                        — compressor knob (0-10)
Strip[0].Gate                        — gate knob (0-10)
Strip[0].Denoiser                    — denoiser knob (0-10)
Strip[0].Reverb                      — reverb send (0-10)
Strip[0].Delay                       — delay send (0-10)
Strip[0].EQ.on                       — EQ on/off toggle
Strip[0].Color_x, .Color_y           — Color Panel (fx echo, brightness)
```

### Що НЕ працює через API
- **Strip Parametric EQ bands** (`Strip[0].EQ.channel[0].cell[*]`) — читається, але НЕ записується!
  - API приймає set (ret=0), значення не змінюються
  - Потрібно налаштовувати вручну через GUI
- **FX.Reverb.On / FX.Delay.On** — аналогічно, не записуються через API
  - Потрібно вмикати вручну в Master Section GUI
- **`VBVMR_SetParameterFloat()`** для knob-параметрів — set повертає OK але не застосовує
  - Використовувати `VBVMR_SetParameters()` замість!
- **Macro Buttons hotkeys** — працюють тільки коли вікно MacroButtons у фокусі (обмеження Windows)

### Формат `.knob` vs без `.knob`
- `Strip[0].Comp` = зовнішнє значення (те що бачиш в GUI)
- `Strip[0].Comp.knob` = внутрішнє значення (може відрізнятись)
- Для SET використовувати БЕЗ `.knob`

### Приклад робочого коду
```python
import ctypes, time
vm = ctypes.cdll.LoadLibrary(r"C:\Program Files (x86)\VB\Voicemeeter\VoicemeeterRemote64.dll")
vm.VBVMR_Login()
time.sleep(0.2)

# SET — використовувати SetParameters (macro format)
vm.VBVMR_SetParameters(b"Strip[0].Gain=6;Strip[0].Comp=1;")
time.sleep(0.3)
vm.VBVMR_IsParametersDirty()  # flush
time.sleep(0.1)

# GET — верифікація
v = ctypes.c_float()
vm.VBVMR_GetParameterFloat(b"Strip[0].Gain", ctypes.byref(v))
print(v.value)  # 6.0

vm.VBVMR_Logout()
```
