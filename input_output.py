"""
PushToTalk Input/Output
Keyboard simulation (SendInput), paste strategies, keyboard layout detection
"""

import time
import logging
import ctypes

logger = logging.getLogger("ptt")

# === Low-level SendInput structures ===

SendInput = ctypes.windll.user32.SendInput
PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort),
    ]


class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]


# Virtual key codes
VK_CONTROL = 0x11
VK_SHIFT = 0x10
VK_V = 0x56
VK_RETURN = 0x0D
VK_INSERT = 0x2D

KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002


def press_key(hex_key_code):
    """Press a key via SendInput."""
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(hex_key_code, 0x48, 0, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def release_key(hex_key_code):
    """Release a key via SendInput."""
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(hex_key_code, 0x48, 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def type_unicode_char(char):
    """Type a single Unicode character via SendInput."""
    extra = ctypes.c_ulong(0)
    # Key down
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, ord(char), KEYEVENTF_UNICODE, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    # Key up
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, ord(char), KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def type_text(text, delay=0.002):
    """Type text character by character (for DCV client)."""
    for char in text:
        type_unicode_char(char)
        time.sleep(delay)


# === Window detection ===

# Terminal window title keywords
TERMINAL_TITLES = [
    "dcv", "terminal", "консоль", "cmd", "powershell",
    "bash", "wsl", "ubuntu", "mintty", "putty", "ssh", "wezterm", "dev-ubuntu",
]

# Terminal process names
TERMINAL_PROCESSES = ["wezterm-gui.exe", "windowsterminal.exe", "mintty.exe", "putty.exe"]


def get_active_window_info() -> tuple:
    """Get active window title and process name. Returns (title, process_name)."""
    import win32gui

    try:
        hwnd = win32gui.GetForegroundWindow()
        window_title = win32gui.GetWindowText(hwnd).lower()
    except Exception:
        return "", ""

    process_name = ""
    try:
        import win32process
        import psutil
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        process_name = psutil.Process(pid).name().lower()
    except Exception:
        pass

    return window_title, process_name


def detect_paste_mode(window_title: str, process_name: str) -> str:
    """Detect which paste method to use.

    Returns: 'dcv', 'terminal', or 'standard'
    """
    if "dcv" in window_title:
        return "dcv"

    is_terminal = (
        any(term in window_title for term in TERMINAL_TITLES)
        or any(proc == process_name for proc in TERMINAL_PROCESSES)
    )
    if is_terminal:
        return "terminal"

    return "standard"


def paste_text(text: str, mode: str, auto_enter: bool = False):
    """Paste text using the appropriate method for the active window.

    mode: 'dcv' | 'terminal' | 'standard'
    """
    import pyperclip

    if mode == "dcv":
        logger.debug("DCV char-by-char input")
        type_text(text)
        return

    # Copy to clipboard
    pyperclip.copy(text)
    time.sleep(0.02)
    if pyperclip.paste() != text:
        pyperclip.copy(text)
        time.sleep(0.03)

    if mode == "terminal":
        # Ctrl+Shift+V
        press_key(VK_CONTROL)
        press_key(VK_SHIFT)
        press_key(VK_V)
        time.sleep(0.01)
        release_key(VK_V)
        release_key(VK_SHIFT)
        release_key(VK_CONTROL)
        logger.debug("Ctrl+Shift+V")
    else:
        # Ctrl+V
        press_key(VK_CONTROL)
        press_key(VK_V)
        time.sleep(0.01)
        release_key(VK_V)
        release_key(VK_CONTROL)
        logger.debug("Ctrl+V")

    if auto_enter and mode != "dcv":
        time.sleep(0.01)
        press_key(VK_RETURN)
        release_key(VK_RETURN)


# === Keyboard layout detection ===

class GUITHREADINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_ulong),
        ("flags", ctypes.c_ulong),
        ("hwndActive", ctypes.c_void_p),
        ("hwndFocus", ctypes.c_void_p),
        ("hwndCapture", ctypes.c_void_p),
        ("hwndMenuOwner", ctypes.c_void_p),
        ("hwndMoveSize", ctypes.c_void_p),
        ("hwndCaret", ctypes.c_void_p),
        ("rcCaret", ctypes.c_long * 4),
    ]


def get_keyboard_language() -> str:
    """Detect current keyboard layout language.

    Uses GetGUIThreadInfo to find the window with input focus —
    important for Electron/WebView2 apps (Teams, VS Code, etc.)
    where the child window with focus may be on a different thread.
    """
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    thread_id = user32.GetWindowThreadProcessId(hwnd, None)

    gui_info = GUITHREADINFO()
    gui_info.cbSize = ctypes.sizeof(GUITHREADINFO)
    if user32.GetGUIThreadInfo(thread_id, ctypes.byref(gui_info)):
        focus_hwnd = gui_info.hwndFocus
        if focus_hwnd:
            focus_thread = user32.GetWindowThreadProcessId(int(focus_hwnd), None)
            layout_id = user32.GetKeyboardLayout(focus_thread)
        else:
            layout_id = user32.GetKeyboardLayout(thread_id)
    else:
        layout_id = user32.GetKeyboardLayout(thread_id)

    lang_id = layout_id & 0xFFFF
    # 0x0422 = Ukrainian
    if lang_id == 0x0422:
        return "uk"
    return "en"
