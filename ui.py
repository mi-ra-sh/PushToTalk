"""
PushToTalk UI Components
System tray icon, recording indicator, transcription history
"""

import os
import time
import threading
import logging
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont
import pystray
import pyperclip

from config import LANG_CONFIGS, HOTKEY_NAMES, SOUNDS, AVAILABLE_MODELS, BASE_DIR

logger = logging.getLogger("ptt")


class TrayManager:
    """System tray icon with menu, language display, and transcription history."""

    def __init__(self, config: dict, callbacks: dict):
        """
        callbacks: dict with keys:
            - on_toggle_language: callable
            - on_refresh_model: callable
            - on_exit: callable
            - on_change_hotkey: callable(hotkey_name)
            - on_change_model: callable(model_name)
            - on_toggle_lora: callable
            - get_language: callable -> str
            - get_history: callable -> list[dict]
        """
        self.config = config
        self.callbacks = callbacks
        self._icon = None
        self._early_icon = None
        self._model_updates = None
        self._is_loading = False

    def show_loading(self):
        """Show early loading icon before model loads, with Exit menu."""
        import os as _os
        image = self._create_icon_image("...", loading=True)
        menu = pystray.Menu(
            pystray.MenuItem("Push-to-Talk (loading...)", None, enabled=False),
            pystray.MenuItem("─────────", None, enabled=False),
            pystray.MenuItem("Вихід", lambda icon, item: _os._exit(0)),
        )
        self._icon = pystray.Icon("push_to_talk", image, "", menu)
        thread = threading.Thread(target=self._icon.run, daemon=True)
        thread.start()
        self._early_icon = self._icon

    def setup(self):
        """Update tray icon after model is loaded."""
        lang = self.callbacks["get_language"]()
        label = LANG_CONFIGS[lang]["label"]
        image = self._create_icon_image(label)
        menu = self._build_menu()

        if self._icon:
            self._icon.icon = image
            self._icon.title = ""
            self._icon.menu = menu
        else:
            self._icon = pystray.Icon("push_to_talk", image, "", menu)
            thread = threading.Thread(target=self._icon.run, daemon=True)
            thread.start()

    def update_icon(self):
        """Update icon after language change."""
        if self._icon:
            lang = self.callbacks["get_language"]()
            label = LANG_CONFIGS[lang]["label"]
            self._icon.icon = self._create_icon_image(label)
            self._icon.title = ""

    def update_menu(self):
        """Rebuild menu (e.g. after history changes)."""
        if self._icon:
            self._icon.menu = self._build_menu()

    def stop(self):
        """Stop the tray icon."""
        if self._icon:
            self._icon.stop()

    def _build_menu(self):
        """Build the tray context menu."""
        lang = self.callbacks["get_language"]()
        hotkey_name = HOTKEY_NAMES.get(self.config["hotkey"], self.config["hotkey"])

        # Current model display name
        model_display = AVAILABLE_MODELS.get(
            self.config["model_name"], self.config["model_name"].split("/")[-1]
        )
        if self.config["use_lora"]:
            model_display += " +LoRA"

        items = [
            pystray.MenuItem("Push-to-Talk", None, enabled=False),
            pystray.MenuItem("─────────", None, enabled=False),
            pystray.MenuItem(
                lambda text: f"Мова: {LANG_CONFIGS[self.callbacks['get_language']()]['label']} (Ctrl+Alt+L)",
                lambda icon, item: self._on_toggle_language(),
            ),
        ]

        # Model submenu
        model_items = []
        for model_id, display_name in AVAILABLE_MODELS.items():
            is_current = model_id == self.config["model_name"]
            model_items.append(
                pystray.MenuItem(
                    f"{'● ' if is_current else '  '}{display_name}",
                    self._make_model_callback(model_id),
                )
            )

        # LoRA toggle (only for whisper-large-v3 when adapter exists)
        adapter_dir = os.path.join(BASE_DIR, "whisper_lora_adapter")
        if os.path.isdir(adapter_dir):
            model_items.append(pystray.MenuItem("─────────", None, enabled=False))
            can_lora = self.config["model_name"] == "openai/whisper-large-v3"
            lora_prefix = "● " if self.config["use_lora"] else "  "
            model_items.append(
                pystray.MenuItem(
                    f"{lora_prefix}LoRA адаптер",
                    lambda icon, item: self._on_toggle_lora(),
                    enabled=can_lora,
                )
            )

        items.append(
            pystray.MenuItem(f"Модель: {model_display}", pystray.Menu(*model_items))
        )

        # Hotkey submenu
        hotkey_items = []
        for key_id, key_name in HOTKEY_NAMES.items():
            is_current = key_id == self.config["hotkey"]
            hotkey_items.append(
                pystray.MenuItem(
                    f"{'● ' if is_current else '  '}{key_name}",
                    self._make_hotkey_callback(key_id),
                )
            )
        items.append(pystray.MenuItem(f"Клавіша: {hotkey_name}", pystray.Menu(*hotkey_items)))

        items.append(pystray.MenuItem("─────────", None, enabled=False))

        # History submenu
        history = self.callbacks["get_history"]()
        if history:
            history_items = []
            for entry in reversed(history[-10:]):
                text = entry.get("text", "")
                display = text[:50] + ("..." if len(text) > 50 else "")
                lang_label = entry.get("language", "?").upper()
                conf = entry.get("confidence", 0)
                conf_str = f" ({conf:.0%})" if conf > 0 else ""
                history_items.append(
                    pystray.MenuItem(
                        f"[{lang_label}]{conf_str} {display}",
                        self._make_history_callback(text),
                    )
                )
            items.append(pystray.MenuItem("Історія", pystray.Menu(*history_items)))
        else:
            items.append(pystray.MenuItem("Історія (порожня)", None, enabled=False))

        items.append(pystray.MenuItem("─────────", None, enabled=False))

        # HuggingFace update status
        if self._model_updates is not None:
            update_items = []
            has_any_update = False
            for model_id, info in self._model_updates.items():
                name = AVAILABLE_MODELS.get(model_id, model_id.split("/")[-1])
                if info.get("error"):
                    update_items.append(
                        pystray.MenuItem(f"  {name}: помилка", None, enabled=False)
                    )
                elif info.get("updated"):
                    has_any_update = True
                    modified = info.get("last_modified", "")[:10]
                    update_items.append(
                        pystray.MenuItem(f"  {name}: оновлено {modified}", None, enabled=False)
                    )
                else:
                    modified = info.get("last_modified", "")[:10]
                    update_items.append(
                        pystray.MenuItem(f"  {name}: актуальна ({modified})", None, enabled=False)
                    )

            update_label = "[!] Оновлення HF" if has_any_update else "HF: актуально"
            items.append(
                pystray.MenuItem(update_label, pystray.Menu(*update_items))
            )

        items.extend([
            pystray.MenuItem("Перезавантажити модель", lambda icon, item: self._on_refresh()),
            pystray.MenuItem("Вихід", lambda icon, item: self._on_exit()),
        ])

        return pystray.Menu(*items)

    def show_model_loading(self):
        """Show loading icon while model is switching."""
        self._is_loading = True
        if self._icon:
            self._icon.icon = self._create_icon_image("...", loading=True)

    def set_model_updates(self, updates):
        """Store HuggingFace update check results."""
        self._model_updates = updates

    def _on_toggle_language(self):
        self.callbacks["on_toggle_language"]()
        self.update_icon()

    def _on_refresh(self):
        self.callbacks["on_refresh_model"]()

    def _on_exit(self):
        self.callbacks["on_exit"]()

    def _on_toggle_lora(self):
        if self._is_loading:
            return
        cb = self.callbacks.get("on_toggle_lora")
        if cb:
            cb()

    def _make_model_callback(self, model_name):
        def callback(icon, item):
            if self._is_loading:
                return
            if model_name == self.config["model_name"]:
                return
            cb = self.callbacks.get("on_change_model")
            if cb:
                cb(model_name)
        return callback

    def _make_hotkey_callback(self, hotkey_name):
        def callback(icon, item):
            self.callbacks["on_change_hotkey"](hotkey_name)
        return callback

    def _make_history_callback(self, text):
        def callback(icon, item):
            pyperclip.copy(text)
            logger.info(f"Copied from history: {text[:40]}...")
        return callback

    @staticmethod
    def _create_icon_image(label, loading=False):
        """Create 256x256 tray icon with language label."""
        size = 256
        image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        bg_color = "#4a3a1e" if loading else "#1a1a2e"
        draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=40, fill=bg_color)

        try:
            font = ImageFont.truetype("arialbd.ttf", 180)
        except OSError:
            try:
                font = ImageFont.truetype("arial.ttf", 180)
            except OSError:
                font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = (size - tw) // 2
        ty = (size - th) // 2 - bbox[1]

        outline_color = "#000000"
        for dx, dy in [(-4, 0), (4, 0), (0, -4), (0, 4), (-3, -3), (3, -3), (-3, 3), (3, 3)]:
            draw.text((tx + dx, ty + dy), label, fill=outline_color, font=font)

        text_color = "#FFD700" if loading else "#FFFFFF"
        draw.text((tx, ty), label, fill=text_color, font=font)

        return image


class RecordingIndicator:
    """Visual recording indicator near cursor with pulsing animation."""

    def __init__(self):
        self.root = None
        self.canvas = None
        self.inner_circle = None
        self.outer_glow = None
        self.is_visible = False
        self.pulse_job = None
        self.pulse_state = 0
        self.current_color = "#00CC00"
        self._thread = threading.Thread(target=self._run_tk, daemon=True)
        self._thread.start()
        time.sleep(0.1)

    def _run_tk(self):
        """Main Tk loop."""
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", "#010101")

        size = 44
        self.root.geometry(f"{size}x{size}")

        self.canvas = tk.Canvas(self.root, width=size, height=size, bg="#010101", highlightthickness=0)
        self.canvas.pack()

        self.outer_glow = self.canvas.create_oval(2, 2, size - 2, size - 2, fill="#44FF44", outline="#66FF66", width=2)

        padding = 8
        self.inner_circle = self.canvas.create_oval(
            padding, padding, size - padding, size - padding, fill="#00CC00", outline="#FFFFFF", width=2
        )

        cx, cy = size // 2, size // 2
        self.canvas.create_oval(cx - 5, cy - 10, cx + 5, cy + 2, fill="white", outline="white")
        self.canvas.create_line(cx, cy + 2, cx, cy + 8, fill="white", width=2)
        self.canvas.create_arc(cx - 8, cy - 2, cx + 8, cy + 12, start=0, extent=-180, style="arc", outline="white", width=2)

        self.root.mainloop()

    def _pulse(self):
        """Pulsing animation using current color."""
        if not self.is_visible or not self.canvas:
            return

        self.pulse_state = (self.pulse_state + 1) % 10

        colors = {
            "#00CC00": (("#00DD00", "#55FF55"), ("#00BB00", "#33DD33")),
            "#FFCC00": (("#FFDD00", "#FFFF55"), ("#EEBB00", "#FFEE33")),
        }
        pair = colors.get(self.current_color, (("#FF0000", "#FF4444"), ("#CC0000", "#FF2222")))
        color, glow = pair[0] if self.pulse_state < 5 else pair[1]

        try:
            self.canvas.itemconfig(self.inner_circle, fill=color)
            self.canvas.itemconfig(self.outer_glow, fill=glow)
        except Exception:
            pass

        if self.is_visible:
            self.pulse_job = self.root.after(100, self._pulse)

    def show(self):
        """Show indicator near cursor."""
        if self.root and not self.is_visible:
            def _show():
                try:
                    x, y = self.root.winfo_pointerxy()
                    self.root.geometry(f"+{x + 15}+{y + 15}")
                    self.root.deiconify()
                    self.is_visible = True
                    self._pulse()
                except Exception:
                    pass
            self.root.after(0, _show)

    def hide(self):
        """Hide indicator."""
        if self.root and self.is_visible:
            def _hide():
                try:
                    if self.pulse_job:
                        self.root.after_cancel(self.pulse_job)
                        self.pulse_job = None
                    self.root.withdraw()
                    self.is_visible = False
                except Exception:
                    pass
            self.root.after(0, _hide)

    def update_status(self, color="#00CC00"):
        """Update color: green=recording, yellow=processing, red=error."""
        self.current_color = color
        if self.root and self.canvas:
            def _update():
                try:
                    self.canvas.itemconfig(self.inner_circle, fill=color)
                    glow_map = {"#00CC00": "#44FF44", "#FFCC00": "#FFEE44"}
                    self.canvas.itemconfig(self.outer_glow, fill=glow_map.get(color, "#FF4444"))
                except Exception:
                    pass
            self.root.after(0, _update)
