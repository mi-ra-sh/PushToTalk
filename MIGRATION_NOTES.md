# Migration: HuggingFace Whisper large-v3 → faster-whisper + large-v3-turbo

**Date:** 2026-04-24
**Before tag:** `pre-faster-whisper-migration` (commit `3f50a88`)

## Why

The HuggingFace `transformers` pathway was the default runtime even though
`faster-whisper` (CTranslate2) was already wired up in a secondary branch.
CT2 is 2–4× faster on the same hardware and frees ~1.4 GB VRAM on the turbo
tier. The large-v3-turbo distilled model retains V3 quality on short,
single-speaker push-to-talk utterances. Net effect: the non-LoRA happy path
becomes cheaper and snappier, and the HF dependency survives only where the
trained adapter requires it.

## Scope

Four files touched — tightly scoped. No other modules edited.

- `config.py`         — model registry, default, legacy config migration.
- `engines/whisper_engine.py` — HF engine reduced to LoRA-only path.
- `push_to_talk.py`   — LoRA toggle re-wired, `use_lora` invariant tightened.
- `ui.py`             — tray colors, LoRA menu gating.
- `MIGRATION_NOTES.md` (new).

`engines/faster_whisper_engine.py`, `engines/__init__.py`, and the rest of
the application were not modified.

## Model registry after migration

| id                   | backend        | model                       | LoRA |
| -------------------- | -------------- | --------------------------- | ---- |
| `whisper-turbo-fast` | faster-whisper | `large-v3-turbo`            | —    |
| `whisper-v3-fast`    | faster-whisper | `large-v3`                  | —    |
| `whisper-v3`         | HF transformers| `openai/whisper-large-v3`   | ✔    |

Default = `whisper-turbo-fast`.

Removed: `whisper-v3-turbo` (HF). It was strictly dominated by the CT2 turbo
build — same weights, heavier inference.

## LoRA adapter flow

The adapter at `whisper_lora_adapter/` was trained against
`openai/whisper-large-v3` (r=32, α=64, 501 user samples) and is not directly
usable by CTranslate2. The HF engine is retained solely for this path and
will now:

- Refuse to load without an adapter on disk (`FileNotFoundError` at load
  time — fail fast, no silent fallback).
- Always merge the adapter and unload it from the PEFT wrapper.

`config["use_lora"]` becomes a derived invariant: `True` iff
`selected_model == "whisper-v3"`. Both `load_config` and `switch_engine`
enforce this, so the toggle in the tray menu is the single source of intent.
Toggling LoRA off flips the model to `whisper-v3-fast` (CT2 base), toggling
it back on flips to `whisper-v3`. One click, one reload.

The tray model submenu hides the HF+LoRA entry when the adapter directory
is missing.

## Legacy config migration

`load_config` now handles two retired identifiers transparently:

- `model_name: "openai/whisper-large-v3-turbo"` → `whisper-turbo-fast`
- `model_name: "openai/whisper-large-v3"` → `whisper-v3` (LoRA) or
  `whisper-v3-fast` depending on the previous `use_lora` flag.
- `selected_model: "whisper-v3-turbo"` → `whisper-turbo-fast`.
- `selected_model: "whisper-v3"` with `use_lora=False` → `whisper-v3-fast`.

Unknown ids fall back to `DEFAULT_MODEL`.

## Rollback

```bash
git reset --hard pre-faster-whisper-migration
```

or cherry-pick the inverse of the migration commit. No data files were
rewritten; `config.json` is self-migrating on next load.

## VRAM savings

Default happy-path VRAM (fp16, GPU load):

- Before: HF `whisper-large-v3` ≈ 3200 MB.
- After:  CT2 `large-v3-turbo`  ≈ 1800 MB.

Δ ≈ **-1400 MB (~44%)** freed on the GPU for the default path. LoRA path
is unchanged at ~3200 MB.
