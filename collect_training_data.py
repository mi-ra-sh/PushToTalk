"""
GUI для збору тренувальних даних для fine-tuning Whisper.
Показує речення → записує голос → зберігає WAV 16kHz + metadata.csv
"""

import sys
import os
import csv
import time
import wave
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import sounddevice as sd

# Fix Windows console encoding
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

# ============ НАЛАШТУВАННЯ ============
SAMPLE_RATE = 16000
CHANNELS = 1
INPUT_DEVICE = "Voicemeeter Out B1"
SENTENCES_FILE = os.path.join(os.path.dirname(__file__), "training_sentences.txt")
DATA_DIR = os.path.join(os.path.dirname(__file__), "training_data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv")
# ======================================


def find_device():
    """Знайти аудіо пристрій за назвою"""
    for i, d in enumerate(sd.query_devices()):
        if INPUT_DEVICE in d['name'] and d['max_input_channels'] > 0:
            return i
    return None


def load_sentences(filepath):
    """Завантажити речення з файлу (ігноруючи коментарі та порожні рядки)"""
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('='):
                continue
            # Формат: CATEGORY | sentence
            if '|' in line:
                parts = line.split('|', 1)
                category = parts[0].strip()
                text = parts[1].strip()
                sentences.append((category, text))
    return sentences


def load_recorded_indices(metadata_path):
    """Завантажити індекси вже записаних речень"""
    recorded = set()
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='|')
            for row in reader:
                source = row.get('source', '')
                if source.startswith('sentence_'):
                    try:
                        idx = int(source.split('_')[1])
                        recorded.add(idx)
                    except (ValueError, IndexError):
                        pass
    return recorded


def get_next_file_id(metadata_path):
    """Отримати наступний ID для файлу"""
    max_id = 0
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='|')
            for row in reader:
                fname = row.get('file_name', '')
                try:
                    fid = int(fname.replace('.wav', ''))
                    max_id = max(max_id, fid)
                except ValueError:
                    pass
    return max_id + 1


class TrainingDataCollector:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Training Data Collector")
        self.root.geometry("900x700")
        self.root.configure(bg='#1e1e2e')

        # Стиль
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#1e1e2e')
        style.configure('TLabel', background='#1e1e2e', foreground='#cdd6f4', font=('Segoe UI', 11))
        style.configure('Title.TLabel', background='#1e1e2e', foreground='#89b4fa', font=('Segoe UI', 16, 'bold'))
        style.configure('Sentence.TLabel', background='#313244', foreground='#f5e0dc', font=('Segoe UI', 14))
        style.configure('Category.TLabel', background='#1e1e2e', foreground='#a6e3a1', font=('Segoe UI', 10))
        style.configure('Status.TLabel', background='#1e1e2e', foreground='#f9e2af', font=('Segoe UI', 10))
        style.configure('TButton', font=('Segoe UI', 11))
        style.configure('Record.TButton', font=('Segoe UI', 13, 'bold'))
        style.configure('TProgressbar', troughcolor='#313244', background='#89b4fa')

        # Дані
        self.sentences = load_sentences(SENTENCES_FILE)
        self.recorded_indices = load_recorded_indices(METADATA_FILE)
        self.next_file_id = get_next_file_id(METADATA_FILE)
        self.current_index = self._find_next_unrecorded(0)
        self.is_recording = False
        self.recorded_audio = []
        self.stream = None
        self.device_index = find_device()
        self.mode = 'sentences'  # 'sentences' або 'free'

        # Створити директорії
        os.makedirs(AUDIO_DIR, exist_ok=True)

        # Ініціалізувати metadata.csv якщо не існує
        if not os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, delimiter='|')
                writer.writerow(['file_name', 'transcription', 'category', 'source', 'duration'])

        self._build_ui()
        self._update_display()

    def _find_next_unrecorded(self, start):
        """Знайти наступне незаписане речення"""
        for i in range(start, len(self.sentences)):
            if i not in self.recorded_indices:
                return i
        # Якщо всі записані, повернути start
        return start

    def _build_ui(self):
        """Побудувати інтерфейс"""
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        # Заголовок + режим
        top_frame = ttk.Frame(main)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(top_frame, text="Whisper Training Data", style='Title.TLabel').pack(side=tk.LEFT)

        mode_frame = ttk.Frame(top_frame)
        mode_frame.pack(side=tk.RIGHT)
        self.mode_var = tk.StringVar(value='sentences')
        ttk.Radiobutton(mode_frame, text="Речення", variable=self.mode_var, value='sentences',
                        command=self._switch_mode).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="Вільний запис", variable=self.mode_var, value='free',
                        command=self._switch_mode).pack(side=tk.LEFT, padx=5)

        # Прогрес
        progress_frame = ttk.Frame(main)
        progress_frame.pack(fill=tk.X, pady=(0, 10))

        self.progress_label = ttk.Label(progress_frame, text="", style='Status.TLabel')
        self.progress_label.pack(side=tk.LEFT)

        self.progress_bar = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))

        # Категорія
        self.category_label = ttk.Label(main, text="", style='Category.TLabel')
        self.category_label.pack(anchor=tk.W, pady=(5, 2))

        # Речення (у рамці)
        sentence_frame = tk.Frame(main, bg='#313244', padx=15, pady=15)
        sentence_frame.pack(fill=tk.X, pady=(0, 10))

        self.sentence_label = tk.Label(
            sentence_frame, text="", bg='#313244', fg='#f5e0dc',
            font=('Segoe UI', 14), wraplength=820, justify=tk.LEFT, anchor=tk.W
        )
        self.sentence_label.pack(fill=tk.X)

        # Вільний текст (для режиму free)
        self.free_frame = ttk.Frame(main)
        self.free_text = scrolledtext.ScrolledText(
            self.free_frame, height=3, font=('Segoe UI', 12),
            bg='#313244', fg='#f5e0dc', insertbackground='white'
        )
        self.free_text.pack(fill=tk.X)
        ttk.Label(self.free_frame, text="Введіть правильний текст після запису:", style='Status.TLabel').pack(
            anchor=tk.W, pady=(0, 3), before=self.free_text)

        # Навігація
        nav_frame = ttk.Frame(main)
        nav_frame.pack(fill=tk.X, pady=(0, 10))

        self.prev_btn = ttk.Button(nav_frame, text="< Попереднє", command=self._prev_sentence)
        self.prev_btn.pack(side=tk.LEFT)

        self.index_label = ttk.Label(nav_frame, text="")
        self.index_label.pack(side=tk.LEFT, padx=20)

        self.next_btn = ttk.Button(nav_frame, text="Наступне >", command=self._next_sentence)
        self.next_btn.pack(side=tk.LEFT)

        self.skip_btn = ttk.Button(nav_frame, text="Пропустити незаписані >>", command=self._skip_to_unrecorded)
        self.skip_btn.pack(side=tk.RIGHT)

        # Кнопки запису
        rec_frame = ttk.Frame(main)
        rec_frame.pack(fill=tk.X, pady=(10, 5))

        self.record_btn = tk.Button(
            rec_frame, text="Натисни та тримай для запису (або Space)",
            bg='#f38ba8', fg='white', font=('Segoe UI', 13, 'bold'),
            activebackground='#eba0ac', activeforeground='white',
            relief=tk.FLAT, padx=20, pady=10
        )
        self.record_btn.pack(fill=tk.X)
        self.record_btn.bind('<ButtonPress-1>', self._start_recording)
        self.record_btn.bind('<ButtonRelease-1>', self._stop_recording)

        # Space для запису
        self.root.bind('<KeyPress-space>', self._start_recording)
        self.root.bind('<KeyRelease-space>', self._stop_recording)

        # Ctrl+P для прослуховування
        self.root.bind('<Control-p>', lambda e: self._playback())

        # Кнопки дій
        action_frame = ttk.Frame(main)
        action_frame.pack(fill=tk.X, pady=(5, 5))

        self.play_btn = ttk.Button(action_frame, text="Прослухати (Ctrl+P)", command=self._playback)
        self.play_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.save_btn = ttk.Button(action_frame, text="Зберегти (Enter)", command=self._save_recording)
        self.save_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.root.bind('<Return>', lambda e: self._save_recording())

        self.rerecord_btn = ttk.Button(action_frame, text="Перезаписати (R)", command=self._clear_recording)
        self.rerecord_btn.pack(side=tk.LEFT)
        self.root.bind('r', lambda e: self._clear_recording() if not self.free_text.focus_get() == self.free_text else None)

        # Статус
        self.status_label = ttk.Label(main, text="", style='Status.TLabel')
        self.status_label.pack(fill=tk.X, pady=(10, 0))

        # Інформація про пристрій
        device_text = f"Мікрофон: {INPUT_DEVICE}" + (" (знайдено)" if self.device_index is not None else " (НЕ ЗНАЙДЕНО)")
        self.device_label = ttk.Label(main, text=device_text, style='Status.TLabel')
        self.device_label.pack(anchor=tk.W, pady=(5, 0))

        # Статистика
        stats_frame = ttk.Frame(main)
        stats_frame.pack(fill=tk.X, pady=(5, 0))
        self.stats_label = ttk.Label(stats_frame, text="", style='Status.TLabel')
        self.stats_label.pack(anchor=tk.W)

        self._update_stats()

    def _switch_mode(self):
        """Перемкнути режим"""
        self.mode = self.mode_var.get()
        if self.mode == 'free':
            self.free_frame.pack(fill=tk.X, pady=(0, 10), after=self.category_label.master.winfo_children()[3])
            self.sentence_label.config(text="Говоріть що завгодно, потім введіть правильний текст нижче")
            self.category_label.config(text="[FREE] Вільний запис")
            self.prev_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.DISABLED)
            self.skip_btn.config(state=tk.DISABLED)
        else:
            self.free_frame.pack_forget()
            self.prev_btn.config(state=tk.NORMAL)
            self.next_btn.config(state=tk.NORMAL)
            self.skip_btn.config(state=tk.NORMAL)
            self._update_display()

    def _update_display(self):
        """Оновити відображення поточного речення"""
        if self.mode == 'free':
            return

        if not self.sentences:
            self.sentence_label.config(text="Речення не знайдені!")
            return

        cat, text = self.sentences[self.current_index]
        self.sentence_label.config(text=text)
        self.category_label.config(text=f"[{cat}] — Категорія")

        recorded_mark = " (записано)" if self.current_index in self.recorded_indices else ""
        self.index_label.config(text=f"{self.current_index + 1} / {len(self.sentences)}{recorded_mark}")

        # Прогрес
        total = len(self.sentences)
        done = len(self.recorded_indices)
        self.progress_bar['maximum'] = total
        self.progress_bar['value'] = done
        self.progress_label.config(text=f"Записано: {done} / {total} ({done*100//total if total else 0}%)")

    def _update_stats(self):
        """Оновити статистику"""
        total_files = len([f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')]) if os.path.exists(AUDIO_DIR) else 0
        total_duration = 0
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='|')
                for row in reader:
                    try:
                        total_duration += float(row.get('duration', 0))
                    except ValueError:
                        pass
        mins = total_duration / 60
        self.stats_label.config(text=f"Файлів: {total_files} | Загальна тривалість: {mins:.1f} хв")

    def _prev_sentence(self):
        if self.current_index > 0:
            self.current_index -= 1
            self._update_display()
            self._clear_recording()

    def _next_sentence(self):
        if self.current_index < len(self.sentences) - 1:
            self.current_index += 1
            self._update_display()
            self._clear_recording()

    def _skip_to_unrecorded(self):
        next_idx = self._find_next_unrecorded(self.current_index + 1)
        if next_idx >= len(self.sentences):
            # Всі речення записані
            self.status_label.config(text="Всі речення записані!")
            self._update_stats()
            return
        if next_idx != self.current_index:
            self.current_index = next_idx
            self._update_display()
            self._clear_recording()

    def _start_recording(self, event=None):
        if self.is_recording:
            return

        self.is_recording = True
        self.recorded_audio = []
        self.record_btn.config(bg='#a6e3a1', text="Запис...")
        self.status_label.config(text="Записую... (відпустіть для зупинки)")

        def audio_callback(indata, frames, time_info, status):
            if self.is_recording:
                self.recorded_audio.append(indata.copy().flatten())

        try:
            self.stream = sd.InputStream(
                callback=audio_callback,
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                blocksize=4096,
                dtype='float32',
                device=self.device_index
            )
            self.stream.start()
        except Exception as e:
            self.status_label.config(text=f"Помилка: {e}")
            self.is_recording = False
            self.record_btn.config(bg='#f38ba8', text="Натисни та тримай для запису (або Space)")

    def _stop_recording(self, event=None):
        if not self.is_recording:
            return

        # Затримка 0.5с щоб дописати хвіст мовлення (латентність Voicemeeter + останній склад)
        self.record_btn.config(bg='#f9e2af', text="Дописую хвіст...")
        self.root.after(500, self._finalize_recording)

    def _finalize_recording(self):
        self.is_recording = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        self.record_btn.config(bg='#f38ba8', text="Натисни та тримай для запису (або Space)")

        if self.recorded_audio:
            audio = np.concatenate(self.recorded_audio)
            duration = len(audio) / SAMPLE_RATE
            rms = np.sqrt(np.mean(audio ** 2))
            self.status_label.config(
                text=f"Записано {duration:.1f}с (RMS: {rms:.4f}) — Enter=зберегти, Ctrl+P=прослухати, R=перезаписати"
            )
        else:
            self.status_label.config(text="Немає аудіо для збереження")

    def _playback(self):
        """Прослухати записане аудіо"""
        if not self.recorded_audio:
            self.status_label.config(text="Немає аудіо для прослуховування")
            return

        audio = np.concatenate(self.recorded_audio)
        self.status_label.config(text="Відтворення...")

        def _play():
            sd.play(audio, SAMPLE_RATE)
            sd.wait()
            self.root.after(0, lambda: self.status_label.config(
                text="Готово. Enter=зберегти, R=перезаписати"))

        threading.Thread(target=_play, daemon=True).start()

    def _clear_recording(self):
        """Очистити запис"""
        self.recorded_audio = []
        self.status_label.config(text="Запис очищено, готовий до нового запису")

    def _save_recording(self):
        """Зберегти запис"""
        if not self.recorded_audio:
            self.status_label.config(text="Немає аудіо для збереження!")
            return

        audio = np.concatenate(self.recorded_audio).astype(np.float32)
        duration = len(audio) / SAMPLE_RATE

        # Визначити текст і категорію
        if self.mode == 'free':
            text = self.free_text.get("1.0", tk.END).strip()
            if not text:
                self.status_label.config(text="Введіть текст перед збереженням!")
                return
            category = "FREE"
            source = f"free_{self.next_file_id}"
        else:
            category, text = self.sentences[self.current_index]
            source = f"sentence_{self.current_index}"

        # Зберегти WAV
        file_name = f"{self.next_file_id:04d}.wav"
        file_path = os.path.join(AUDIO_DIR, file_name)

        # Нормалізація до -1dB headroom
        peak = np.max(np.abs(audio))
        if peak > 0:
            target = 10 ** (-1 / 20)  # ~0.891
            audio = audio * (target / peak)

        # Конвертація в int16 для WAV
        audio_int16 = (audio * 32767).astype(np.int16)
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())

        # Додати до metadata.csv
        with open(METADATA_FILE, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='|')
            writer.writerow([file_name, text, category, source, f"{duration:.2f}"])

        self.status_label.config(text=f"Збережено: {file_name} ({duration:.1f}с)")
        self.next_file_id += 1

        # Оновити стан
        if self.mode == 'sentences':
            self.recorded_indices.add(self.current_index)
            # Перейти до наступного незаписаного
            self._skip_to_unrecorded()
        else:
            self.free_text.delete("1.0", tk.END)

        self.recorded_audio = []
        self._update_display()
        self._update_stats()


def main():
    if not os.path.exists(SENTENCES_FILE):
        print(f"Файл речень не знайдено: {SENTENCES_FILE}")
        sys.exit(1)

    root = tk.Tk()
    app = TrainingDataCollector(root)
    root.mainloop()


if __name__ == "__main__":
    main()
