# PushToTalk

Windows Push-to-Talk with OpenAI Whisper large-v3 on GPU.

Hold a hotkey, speak, release — transcribed text is auto-pasted into the active window. Optimized for Ukrainian with English code-switching (tech terms stay in Latin script).

## How it works

```
Hold Right Ctrl → Record audio → Release → Whisper transcribes → Text pasted
```

- **Model**: `openai/whisper-large-v3` via HuggingFace Transformers
- **GPU**: Permanent fp16 on CUDA (~3 GB VRAM)
- **LoRA**: Optional fine-tuning adapter auto-loads from `whisper_lora_adapter/`
- **Latency**: ~1-2s for typical utterances

## Features

- **Hold-to-record** with configurable hotkey (Right Ctrl, Scroll Lock, F13-F20)
- **Auto-paste** into any app — typed character-by-character for DCV/remote desktop
- **Keyboard layout sync** — switches Whisper language based on active layout (UK/EN)
- **Code-switching** — Ukrainian speech with English terms: "зроби git push" → correct output
- **Visual indicator** — microphone icon near cursor while recording
- **System tray** — runs in background, right-click menu
- **Anti-hallucination** — silence detection, repeated phrase filtering, RMS threshold
- **Audio processing** — trailing silence trim, internal pause compression, normalization
- **80+ auto-replacements** — common ASR errors, Russian→Ukrainian corrections
- **Sound feedback** — distinct beeps for start/stop/success/error

## Architecture

```
push_to_talk.py     — entry point, tray icon, hotkey listener
├── config.py       — config.json management, logging
├── audio_engine.py — recording, normalization, silence processing
├── whisper_engine.py — model loading, inference, GPU management
├── text_processing.py — replacements, hallucination detection
├── input_output.py — keyboard paste, DCV support, layout detection
└── ui.py           — visual recording indicator
```

## Requirements

- Windows 10/11
- Python 3.11
- NVIDIA GPU with ~3 GB free VRAM
- Microphone

## Setup

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft sounddevice numpy pystray pillow pyperclip psutil
```

```bash
python push_to_talk.py
```

Or use `push_to_talk_silent.vbs` for silent background startup (no console window).

## Fine-tuning (optional)

LoRA fine-tuning on your own voice for better accuracy:

1. `python collect_training_data.py` — record 500 training sentences
2. `python finetune_whisper.py` — train LoRA adapter (rank=32, ~60 MB)
3. `python evaluate_model.py` — compare WER: base vs fine-tuned
4. Restart PushToTalk — adapter auto-loads from `whisper_lora_adapter/`

## Configuration

Settings in `config.json`:

| Setting | Default | Description |
|---------|---------|-------------|
| `hotkey` | `ctrl_r` | Record hotkey |
| `language` | `en` | Whisper language |
| `auto_paste` | `true` | Auto-paste after transcription |
| `beam_size` | `5` | Whisper beam search width |
| `sync_lang_with_keyboard` | `true` | Match Whisper language to keyboard layout |
| `max_recording_seconds` | `120` | Max recording duration |
| `collect_organic_data` | `false` | Save recordings for future fine-tuning |

## License

MIT
