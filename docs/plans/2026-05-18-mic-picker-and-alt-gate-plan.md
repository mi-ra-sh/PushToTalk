# Tray Mic Picker + Alt-Gate Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tray submenu for picking the audio input device with live stream restart, and stop a single Alt+Shift layout switch from silently disabling Push-to-Talk.

**Architecture:** Two independent concerns in adjacent code. (1) Mic picker: deduplicate `sd.query_devices()` results by host-API priority (WASAPI > WDM-KS > DirectSound > MME), expose as a `pystray` submenu, route clicks through a new `AudioRecorder.set_device()` method that stops/restarts the InputStream live. (2) Alt-gate: replace the stale-prone `_alt_held` Python flag in `on_press`'s start-recording branch with a `GetAsyncKeyState` Win32 query at the press moment.

**Tech Stack:** Python 3.11, `sounddevice` (PortAudio), `pystray`, `pynput`, `ctypes` (Win32 user32).

**Spec:** `docs/specs/2026-05-18-mic-picker-and-alt-gate.md`

**Testing note:** Project has no test framework. Verification is manual; each task ends with a concrete repro command the engineer runs before committing.

**Git scope note:** Repo has unrelated dirty files (`config.json`, `config.py`, `push_to_talk.py`, `model_versions.json`, `push_to_talk.bat`, `push_to_talk_silent.vbs`, `run_debug.bat`, `engines/faster_whisper_engine.py`). Stage **only the files listed in each task** — never `git add -A`. If a task modifies one of the pre-dirty files, the engineer must inspect `git diff <file>` before staging to make sure they are not bundling unrelated work.

---

### Task 1: Add device enumeration helper to `audio_engine.py`

**Files:**
- Modify: `C:\PushToTalk\audio_engine.py` (top-level, alongside existing module-level constants)

- [ ] **Step 1: Add HOST_PRIORITY constant and helper functions**

Add at module top (after the `_vad_utils = None` block, before `def _load_vad()`):

```python
# === Input device enumeration ===

HOST_PRIORITY = ["Windows WASAPI", "Windows WDM-KS", "Windows DirectSound", "MME"]
PSEUDO_DEVICE_PREFIXES = ("Microsoft Sound Mapper", "Primary Sound Capture Driver")


def _normalize_device_name(name: str) -> str:
    """Group-key for deduplicating the same physical device across host APIs."""
    import re
    name = name.lower().strip()
    # Drop trailing parenthesized qualifiers like " (USB Audio)"
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name)
    return name


def list_input_devices():
    """Return deduplicated input devices, one entry per real device.

    Returns list of dicts: {index, name, host} sorted by best-host priority
    then device index. Pseudo-devices (Sound Mapper, Primary Capture Driver)
    are filtered out — the synthetic "System default" tray entry covers them.
    """
    groups = {}  # normalized_name -> (priority, index, name, host)
    for idx, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] <= 0:
            continue
        name = d["name"]
        if name.startswith(PSEUDO_DEVICE_PREFIXES):
            continue
        host = sd.query_hostapis(d["hostapi"])["name"]
        if host not in HOST_PRIORITY:
            continue
        prio = HOST_PRIORITY.index(host)
        key = _normalize_device_name(name)
        prev = groups.get(key)
        if prev is None or prio < prev[0]:
            groups[key] = (prio, idx, name, host)

    devices = [
        {"index": idx, "name": name, "host": host}
        for _, idx, name, host in groups.values()
    ]
    devices.sort(key=lambda d: (HOST_PRIORITY.index(d["host"]), d["index"]))
    return devices
```

- [ ] **Step 2: Verify with a one-shot script**

Run from `C:\PushToTalk`:

```bash
C:/Users/Mike/AppData/Local/Programs/Python/Python311/python.exe -c "from audio_engine import list_input_devices; [print(f\"{d['host']:20} | {d['name']}\") for d in list_input_devices()]"
```

Expected output (Mike's machine, May 2026): exactly two rows (Focusrite WASAPI, Realtek WDM-KS), no Sound Mapper, no Primary Capture Driver, no MME duplicate of Focusrite.

```
Windows WASAPI       | Analogue 1 + 2 (Focusrite USB Audio)
Windows WDM-KS       | Microphone (Realtek USB2.0 Audio)
```

- [ ] **Step 3: Commit**

```bash
git -C /c/PushToTalk add audio_engine.py
git -C /c/PushToTalk commit -m "feat(audio): list_input_devices() with host-API dedup"
```

(Confirm `git diff --cached --stat` shows only `audio_engine.py` before committing.)

---

### Task 2: Add `set_device()` to `AudioRecorder`

**Files:**
- Modify: `C:\PushToTalk\audio_engine.py` (inside `AudioRecorder` class, after `stop_stream`)

- [ ] **Step 1: Add the method**

Insert after `stop_stream` (around line 144):

```python
    def set_device(self, device_name):
        """Switch input device live: stop stream, re-resolve index, restart.

        Returns True on success, False if the new device fails to open
        (in which case it tries to restart the previous device).
        """
        previous_index = self._device_index
        was_running = self._stream is not None

        if was_running:
            self.stop_stream()

        try:
            self._device_index = self._find_device(device_name)
            if was_running:
                self.start_stream()
            return True
        except Exception as e:
            logger.error(f"Cannot open device '{device_name}': {e}")
            self._device_index = previous_index
            if was_running:
                try:
                    self.start_stream()
                except Exception as e2:
                    logger.error(f"Could not restore previous device: {e2}")
            return False
```

- [ ] **Step 2: Verify with a one-shot script**

```bash
C:/Users/Mike/AppData/Local/Programs/Python/Python311/python.exe -c "
from audio_engine import AudioRecorder
r = AudioRecorder()
r.start_stream()
print('Initial:', r._device_index)
ok = r.set_device('Focusrite')
print('After Focusrite switch:', ok, r._device_index)
ok = r.set_device('NotARealDevice')
print('After bad switch:', ok, r._device_index)
r.stop_stream()
print('OK')
"
```

Expected: first switch returns `True` and picks a Focusrite index (likely 9 = WASAPI). Bad switch returns `True` too because `_find_device` falls back to `None`/default when name not found — that is current behaviour. The key assertion is: no exception, `stop_stream`/`start_stream` cycles cleanly, "OK" printed at end.

- [ ] **Step 3: Commit**

```bash
git -C /c/PushToTalk add audio_engine.py
git -C /c/PushToTalk commit -m "feat(audio): AudioRecorder.set_device() for live device switch"
```

---

### Task 3: Add `on_change_device` callback in `push_to_talk.py`

**Files:**
- Modify: `C:\PushToTalk\push_to_talk.py` (add callback near other on_change_* handlers, wire into tray.callbacks)

- [ ] **Step 1: Add the callback function**

Insert after `on_change_max_duration` (around line 411, before `# Wire up tray callbacks`):

```python
def on_change_device(device_name):
    """Switch input device live and persist to config."""
    if device_name == "":
        device_name = None
    target = device_name if device_name else "default"
    logger.info(f"Switching input device → {target}")
    ok = recorder.set_device(device_name)
    if ok:
        config["input_device"] = device_name or ""
        save_config(config)
        play_sound("success")
    else:
        play_sound("error")
    tray.update_menu()
```

- [ ] **Step 2: Wire into tray callbacks**

Modify the `tray.callbacks = { ... }` dict (around line 415) — add the new entry:

```python
tray.callbacks = {
    "on_toggle_language": on_toggle_language,
    "on_change_model": on_change_model,
    "on_toggle_lora": on_toggle_lora,
    "on_refresh_model": on_refresh_model,
    "on_exit": on_exit,
    "on_change_hotkey": on_change_hotkey,
    "on_change_max_duration": on_change_max_duration,
    "on_change_device": on_change_device,          # NEW
    "get_language": lambda: state.language,
    "get_history": lambda: state.history,
    "get_engine_status": lambda: state.engine_status,
    "get_input_device": lambda: config.get("input_device", ""),   # NEW
}
```

(The `get_input_device` callback lets the tray show the current selection.)

- [ ] **Step 3: Syntax/import check**

```bash
C:/Users/Mike/AppData/Local/Programs/Python/Python311/python.exe -c "import ast; ast.parse(open(r'C:\PushToTalk\push_to_talk.py', encoding='utf-8').read()); print('syntax ok')"
```

Expected: `syntax ok`.

- [ ] **Step 4: Commit**

```bash
git -C /c/PushToTalk add push_to_talk.py
git -C /c/PushToTalk commit -m "feat(ptt): on_change_device callback for tray device picker"
```

**Caution:** `push_to_talk.py` is in the pre-dirty list. Before staging, run `git diff push_to_talk.py` and confirm the only changes are the new function and the two new dict entries. If older un-committed work appears, stop and resolve manually.

---

### Task 4: Add "Мікрофон" submenu in `ui.py`

**Files:**
- Modify: `C:\PushToTalk\ui.py` (inside `_build_menu`, alongside other submenus; plus a small callback factory)

- [ ] **Step 1: Import the device lister at top of `ui.py`**

Modify the import block near line 15:

```python
from config import LANG_CONFIGS, HOTKEY_NAMES, MODELS, LORA_COMPATIBLE_MODELS, BASE_DIR
from audio_engine import list_input_devices   # NEW
```

- [ ] **Step 2: Add the callback factory**

Insert after `_make_duration_callback` (around line 289):

```python
    def _make_device_callback(self, device_name):
        def callback(icon, item):
            cb = self.callbacks.get("on_change_device")
            if cb:
                cb(device_name)
        return callback
```

- [ ] **Step 3: Build the "Мікрофон" submenu in `_build_menu`**

Insert after the max-recording duration block (after `items.append(pystray.MenuItem(f"Макс. запис: ...`, around line 188) and before the divider `items.append(pystray.MenuItem("─────────", ...))`:

```python
        # Input device submenu
        current_device = self.callbacks.get("get_input_device", lambda: "")()
        try:
            devices = list_input_devices()
        except Exception as e:
            logger.warning(f"list_input_devices failed: {e}")
            devices = []

        def _short_label(name):
            # Keep menu rows readable
            return name[:38] + ("…" if len(name) > 38 else "")

        device_items = []
        for d in devices:
            is_current = bool(current_device) and current_device.lower() in d["name"].lower()
            mark = "● " if is_current else "  "
            host_short = d["host"].replace("Windows ", "")
            device_items.append(
                pystray.MenuItem(
                    f"{mark}{_short_label(d['name'])}  [{host_short}]",
                    self._make_device_callback(d["name"]),
                )
            )

        device_items.append(pystray.MenuItem("─────────", None, enabled=False))
        is_default = not current_device
        device_items.append(
            pystray.MenuItem(
                f"{'● ' if is_default else '  '}System default",
                self._make_device_callback(""),
            )
        )

        # Header label for the parent menu item
        if current_device:
            header_label = _short_label(current_device)
        else:
            header_label = "System default"
        items.append(
            pystray.MenuItem(f"Мікрофон: {header_label}", pystray.Menu(*device_items))
        )
```

- [ ] **Step 4: Syntax check**

```bash
C:/Users/Mike/AppData/Local/Programs/Python/Python311/python.exe -c "import ast; ast.parse(open(r'C:\PushToTalk\ui.py', encoding='utf-8').read()); print('syntax ok')"
```

Expected: `syntax ok`.

- [ ] **Step 5: Commit**

```bash
git -C /c/PushToTalk add ui.py
git -C /c/PushToTalk commit -m "feat(ui): tray Microphone submenu with host-prioritized devices"
```

---

### Task 5: Replace `_alt_held` gate with live `GetAsyncKeyState` query

**Files:**
- Modify: `C:\PushToTalk\push_to_talk.py` (`on_press`, around line 580–611, plus a module-level helper)

- [ ] **Step 1: Add helper near other module-level utilities**

Insert just after `current_hotkey = resolve_hotkey(config["hotkey"])` (around line 335):

```python
def _alt_pressed_now():
    """Live Alt-key state via Win32. Immune to pynput release-event loss
    when Windows consumes Alt-up for layout switching (Alt+Shift)."""
    try:
        import ctypes
        u32 = ctypes.windll.user32
        # VK_LMENU = 0xA4, VK_RMENU = 0xA5
        return bool(u32.GetAsyncKeyState(0xA4) & 0x8000) or \
               bool(u32.GetAsyncKeyState(0xA5) & 0x8000)
    except Exception:
        return False
```

- [ ] **Step 2: Replace the `_alt_held` gate in `on_press`**

Find this block in `on_press` (currently lines 594–611):

```python
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
```

Replace with:

```python
    # Start recording — live Alt check, not stale _alt_held tracked flag
    if key == current_hotkey and not recorder.is_recording:
        if _alt_pressed_now():
            return
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
```

Leave the `_alt_held` updates in `on_press`/`on_release` untouched — the Ctrl+Alt+L combo branch (around line 590) still uses the tracked flag and is correct there (both keys are updated in the same press frame).

- [ ] **Step 3: Syntax + helper sanity check**

```bash
C:/Users/Mike/AppData/Local/Programs/Python/Python311/python.exe -c "
import ast
ast.parse(open(r'C:\PushToTalk\push_to_talk.py', encoding='utf-8').read())
import ctypes
print('alt now:', bool(ctypes.windll.user32.GetAsyncKeyState(0xA4) & 0x8000) or bool(ctypes.windll.user32.GetAsyncKeyState(0xA5) & 0x8000))
print('syntax ok')
"
```

Expected: `alt now: False` (assuming you aren't pressing Alt as the script runs), `syntax ok`.

- [ ] **Step 4: Commit**

```bash
git -C /c/PushToTalk add push_to_talk.py
git -C /c/PushToTalk commit -m "fix(ptt): live Alt check in on_press, immune to layout-switch event loss"
```

---

### Task 6: Restart PTT and verify both fixes end-to-end

**Files:** none — runtime verification only.

- [ ] **Step 1: Stop the running PTT instance**

```powershell
Get-Process pythonw -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*Python311*' } | Stop-Process
```

Expected: exit code 0, no error. Confirm with `Get-Process pythonw -ErrorAction SilentlyContinue` returning nothing.

- [ ] **Step 2: Relaunch via the existing Desktop shortcut**

Either double-click `Push-to-Talk.lnk` on the Desktop, or:

```powershell
& "$env:USERPROFILE\Desktop\Push-to-Talk.lnk"
```

Wait ~10 seconds for model load; confirm tray icon turns from loading state to the UK/EN label.

- [ ] **Step 3: Tail the log during testing**

In a separate terminal:

```powershell
Get-Content C:\PushToTalk\push_to_talk.log -Wait -Tail 10
```

- [ ] **Step 4: Verify mic picker is present**

Open tray menu → confirm "Мікрофон: …" item appears between "Макс. запис" and the divider. Open submenu → confirm: exactly one row for Focusrite with `[WASAPI]` badge, one row for Realtek with `[WDM-KS]` badge, divider, "System default" row at bottom. No Sound Mapper. No MME duplicate.

- [ ] **Step 5: Verify mic picker live-switches**

Click "Focusrite (Analogue 1+2) [WASAPI]". Log must show:

```
Switching input device → Analogue 1 + 2 (Focusrite USB Audio)
Мікрофон: Analogue 1 + 2 (Focusrite USB Audio)
```

(or similar — the second line comes from `_find_device`'s existing log). Then press Right Ctrl, speak briefly, release — must produce a normal `[UK] <text>` log line, not a `[!!]` hallucination and not silence.

- [ ] **Step 6: Verify the Alt-gate fix**

Sequence:
1. Press Right Ctrl, say "перший", release. Expect `[UK] перший...` in log.
2. Press Alt+Shift to switch layout UK→EN.
3. Press Right Ctrl, say "second", release. Expect `[EN] second.` in log.
4. Press Alt+Shift to switch back EN→UK.
5. Press Right Ctrl, say "третій", release. Expect `[UK] третій.` in log.

All three presses must log a `Recording...` line. Before the fix, only the first would.

- [ ] **Step 7: Commit nothing — verification is non-code**

If any step fails, return to the corresponding task (e.g., step 5 fails → recheck Task 2/3/4; step 6 fails → recheck Task 5).

---

## Rollback

If the running app behaves worse after restart:

```bash
git -C /c/PushToTalk log --oneline -10
git -C /c/PushToTalk revert <commit-sha>   # for the offending task's commit
```

Each task is its own commit, so revert granularity matches task granularity. Restart PTT after each revert to confirm.
