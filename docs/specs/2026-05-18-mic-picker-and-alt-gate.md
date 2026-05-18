# Tray mic picker + Alt-gate fix

Date: 2026-05-18
Status: approved (Mike), pending implementation

## Background

Two independent issues observed on the current main:

1. **Stale `input_device` in `config.json`.** Value is `"Stream Mix (Elgato Virtual Audio)"`,
   which no longer exists after the Wave Link removal. `AudioRecorder._find_device`
   logs the warning and falls back to system default — currently the MME 4-channel
   Focusrite (44.1 kHz) instead of the available WASAPI mono variant.

2. **"Only works once" bug.** After process start, the first Right Ctrl press records
   normally; every subsequent press is silently ignored (no `Recording...` log line).
   Root cause: `on_press` in `push_to_talk.py:594` gates the start-recording branch on
   the locally-tracked `_alt_held` flag. When the user switches keyboard layout via
   Alt+Shift (UK↔EN), Windows can swallow the Alt key-up event at the low-level hook
   layer; pynput never delivers the release, `_alt_held` is stuck at `True`, and the
   gate suppresses the hotkey until the next process restart.

The two issues are unrelated. Both are addressed in one change set because they touch
adjacent code paths and ship together cleanly.

## Goal

- User can pick the input device from the tray menu without editing `config.json` or
  restarting the process.
- A single Alt+Shift layout switch no longer disables PTT.

Non-goals: device hot-plug detection, mid-recording device swap, multi-mic mixing,
test-mic playback.

## Design

### Part 1 — Tray mic picker

**UI.** New submenu inserted after "Клавіша" and before "Макс. запис":

```
Мікрофон: <current short name>  ▶
   ●  Focusrite (Analogue 1+2)        [WASAPI]
      Realtek USB2.0 Microphone       [WDM-KS]
   ─────────
      System default
```

- `●` marks the currently active device (or "System default" when `input_device`
  is unset / not found).
- Host API badge is appended after the name.
- Submenu is rebuilt on each open via `_build_menu`; no live hot-plug listener.

**Deduplication.** `sd.query_devices()` typically exposes the same physical mic four
times (one per host API). Group by normalized name and pick the best host:

```python
HOST_PRIORITY = ["Windows WASAPI", "Windows WDM-KS", "Windows DirectSound", "MME"]

def list_input_devices():
    groups = {}  # normalized_name -> (priority, index, raw_name, host_name)
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] <= 0:
            continue
        host = sd.query_hostapis(d["hostapi"])["name"]
        if host not in HOST_PRIORITY:
            continue
        key = normalize_name(d["name"])
        prio = HOST_PRIORITY.index(host)
        prev = groups.get(key)
        if prev is None or prio < prev[0]:
            groups[key] = (prio, i, d["name"], host)
    return [(idx, name, host) for _, idx, name, host in groups.values()]
```

`normalize_name` lowercases, drops trailing parenthesized qualifiers (e.g. `" (USB Audio)"`),
and collapses whitespace. The display name keeps original casing/parens; only the
group key is normalized.

Pseudo-devices (`Microsoft Sound Mapper - Input`, `Primary Sound Capture Driver`) are
excluded by an explicit name-prefix filter — they group under unique names and would
otherwise show up as separate rows. The synthetic "System default" row at the bottom
of the submenu fills the same role with cleaner semantics (it passes `None` to
`_find_device` and lets sounddevice resolve the default).

**Persistence.** `config["input_device"]` stores the chosen sounddevice `name` (string)
or empty string for "System default". The existing `_find_device(name)` substring +
WASAPI-prefer logic is kept — it makes the saved value stable across USB reconnects.

**Live restart.** Add to `AudioRecorder`:

```python
def set_device(self, device_name):
    self.stop_stream()
    try:
        new_idx = self._find_device(device_name)
        self._device_index = new_idx
        self.start_stream()
        return True
    except Exception as e:
        logger.error(f"Cannot open device '{device_name}': {e}")
        # Try to recover with previous device
        try:
            self.start_stream()
        except Exception:
            pass
        return False
```

New tray callback `on_change_device(name)` calls `recorder.set_device(name)`,
saves config on success, plays error sound on failure, rebuilds menu.

### Part 2 — Alt-gate fix

Replace the locally-tracked `_alt_held` gate in the start-recording branch with a
live OS query:

```python
def _alt_pressed_now():
    """Live Alt-key state via Win32, immune to pynput release-event loss."""
    import ctypes
    u32 = ctypes.windll.user32
    return bool(u32.GetAsyncKeyState(0xA4) & 0x8000) or \
           bool(u32.GetAsyncKeyState(0xA5) & 0x8000)  # VK_LMENU, VK_RMENU
```

`on_press` becomes:

```python
def on_press(key):
    global _ctrl_held, _alt_held
    if key in (Key.ctrl_l, Key.ctrl_r):
        _ctrl_held = True
    if key in (Key.alt_l, Key.alt_r, Key.alt_gr):
        _alt_held = True

    # Ctrl+Alt+L — toggle language (uses tracked state, both keys updated this frame)
    if _ctrl_held and _alt_held and hasattr(key, "vk") and key.vk == 76:
        on_toggle_language()
        return

    # Start recording — verify Alt via live OS state, not stale tracked flag
    if key == current_hotkey and not recorder.is_recording:
        if _alt_pressed_now():
            return
        ...
```

`_alt_held` stays around for the Ctrl+Alt+L combo, where both keys are updated within
the same press-frame, so stale state is not an issue there.

Module-level `_alt_pressed_now` is Windows-only. Project already imports
`ctypes.windll.user32` indirectly via dependencies and is Windows-only by design
(WASAPI prefer, Win32 paths, etc.). No new dependency.

### Stretch

If `config["input_device"]` is the old Elgato/Voicemeeter literal string, log it
once at startup with a hint that user can pick a new device from the tray. Avoids
silent fallback confusion for any user with a similarly stale config.

## Risk and verification

**Risks.**

- `sd.stop()` on a running InputStream can block briefly. Acceptable: this is a
  user-initiated UI action, not a hot path.
- Picking a device that doesn't open (e.g., exclusive-mode held by another app) —
  caught by try/except, error sound, fallback to previous.
- `GetAsyncKeyState` returns the asynchronous key state at the moment of call,
  which is exactly what we want: "is Alt physically down right now."

**Verification.**

- Manual: restart PTT, press Right Ctrl 5+ times without speaking → all should log
  `Recording...` and produce `[--] too short` or `[!!]` hallucination, but never
  silently no-op.
- Manual: press Right Ctrl, switch UK↔EN via Alt+Shift, press Right Ctrl again →
  second press must register. Repeat 3 cycles.
- Manual: open tray "Мікрофон" submenu, pick Focusrite → next recording must use
  Focusrite (visible in log when running with debug, and audio captures correctly).
- Manual: pick "System default" → falls back to sounddevice default.

## Out of scope

- Hot-plug detection (device added/removed at runtime). Submenu refreshes on open,
  which is sufficient for the common case.
- Per-device gain/AGC controls.
- Migrating away from pynput.
