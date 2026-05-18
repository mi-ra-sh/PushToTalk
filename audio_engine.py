"""
PushToTalk Audio Engine
Recording, processing, silence handling, VAD, device management
"""

import threading
import logging
import numpy as np
import sounddevice as sd

logger = logging.getLogger("ptt")

# === Silero VAD (lazy-loaded) ===
_vad_model = None
_vad_utils = None


# === Input device enumeration ===

HOST_PRIORITY = ["Windows WASAPI", "Windows WDM-KS", "Windows DirectSound", "MME"]
PSEUDO_DEVICE_PREFIXES = ("Microsoft Sound Mapper", "Primary Sound Capture Driver")


def _normalize_device_name(name: str) -> str:
    """Group-key for deduplicating the same physical device across host APIs.

    Strips everything from the first parenthesis onward — MME truncates
    device names at ~32 chars, often leaving an unbalanced "(...". Matching
    only balanced parens would miss those truncations.
    """
    import re
    name = name.lower().strip()
    name = re.sub(r"\s*\(.*$", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def list_input_devices():
    """Return deduplicated input devices, one entry per real device.

    Returns list of dicts: {index, name, host} sorted by best-host priority
    then device index. Pseudo-devices (Sound Mapper, Primary Capture Driver)
    are filtered out — the synthetic "System default" tray entry covers them.
    """
    groups = {}
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


def _load_vad():
    """Lazy-load silero-vad model."""
    global _vad_model, _vad_utils
    if _vad_model is not None:
        return True
    try:
        import torch
        _vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        _vad_utils = utils
        logger.debug("Silero VAD loaded")
        return True
    except Exception as e:
        logger.debug(f"Silero VAD not available, using RMS fallback: {e}")
        return False


def vad_trim(audio, sample_rate=16000, threshold=0.5):
    """Trim non-speech from start and end using silero-vad.

    Returns trimmed audio, or original if VAD unavailable.
    """
    if not _load_vad():
        return audio

    try:
        import torch
        get_speech_timestamps = _vad_utils[0]
        audio_tensor = torch.FloatTensor(audio)
        timestamps = get_speech_timestamps(audio_tensor, _vad_model, sampling_rate=sample_rate, threshold=threshold)

        if not timestamps:
            return audio  # No speech detected, return as-is (let RMS filter handle)

        # Trim to first..last speech segment with padding
        pad = int(sample_rate * 0.1)  # 100ms padding
        start = max(0, timestamps[0]["start"] - pad)
        end = min(len(audio), timestamps[-1]["end"] + pad)

        trimmed_ms = ((timestamps[0]["start"]) + (len(audio) - timestamps[-1]["end"])) / sample_rate * 1000
        if trimmed_ms > 100:
            logger.debug(f"VAD trimmed {trimmed_ms:.0f}ms non-speech")

        return audio[start:end]
    except Exception as e:
        logger.debug(f"VAD trim error, using original: {e}")
        return audio


class AudioRecorder:
    """Manages audio recording from sounddevice InputStream."""

    def __init__(self, sample_rate=16000, blocksize=4096, device_name=None, max_seconds=120, on_max_reached=None):
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.max_seconds = max_seconds
        self.max_samples = max_seconds * sample_rate
        self.on_max_reached = on_max_reached

        self._recording = []
        self._is_recording = False
        self._max_reached = False
        self._lock = threading.Lock()
        self._stream = None
        self._device_index = self._find_device(device_name)

    def _find_device(self, device_name):
        """Find audio device by name substring, preferring WASAPI host API.

        Pre-existing code matched "WASAPI" against d['name'], but the host
        API is on a separate field — so the preference branch never fired
        and substring matches always returned the first MME entry. Now
        queries query_hostapis to check the host correctly.
        """
        if not device_name:
            return None

        fallback = None
        needle = device_name.lower()
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] <= 0:
                continue
            if needle not in d["name"].lower():
                continue
            host = sd.query_hostapis(d["hostapi"])["name"]
            if "WASAPI" in host:
                logger.info(f"   Мікрофон: {d['name']} [{host}]")
                return i
            elif fallback is None:
                fallback = (i, d["name"], host)

        if fallback is not None:
            i, name, host = fallback
            logger.info(f"   Мікрофон: {name} [{host}]")
            return i

        logger.warning(f"Пристрій '{device_name}' не знайдено, використовую default")
        return None

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback — accumulates audio data."""
        if status:
            logger.debug(f"Audio status: {status}")
        if self._is_recording:
            with self._lock:
                total = sum(len(chunk) for chunk in self._recording)
                if total < self.max_samples:
                    self._recording.append(indata.copy().flatten())
                elif not self._max_reached:
                    self._max_reached = True
                    logger.warning(f"Max recording length ({self.max_seconds}s) reached, stop accumulating")
                    if self.on_max_reached:
                        self.on_max_reached()

    def start_stream(self):
        """Start the audio input stream.

        For WASAPI devices, request auto_convert so PortAudio resamples
        from the device's mixer rate (typically 48 kHz) to our 16 kHz
        target instead of rejecting the open with paInvalidSampleRate.
        """
        extra_settings = None
        if self._device_index is not None:
            try:
                d = sd.query_devices(self._device_index)
                host = sd.query_hostapis(d["hostapi"])["name"]
                if "WASAPI" in host:
                    extra_settings = sd.WasapiSettings(auto_convert=True)
            except Exception:
                pass

        self._stream = sd.InputStream(
            callback=self._audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            dtype="float32",
            device=self._device_index,
            extra_settings=extra_settings,
        )
        self._stream.start()
        logger.debug("Audio stream started")

    def stop_stream(self):
        """Stop the audio input stream."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def set_device(self, device_name):
        """Switch input device live: stop stream, re-resolve index, restart.

        Returns True on success. On failure, tries to restart the previous
        device and returns False.
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

    def start_recording(self):
        """Begin recording audio."""
        with self._lock:
            self._recording = []
            self._is_recording = True
            self._max_reached = False

    def stop_recording(self):
        """Stop recording and return the audio buffer as numpy array, or None if empty."""
        with self._lock:
            self._is_recording = False
            if not self._recording:
                return None
            audio = np.concatenate(self._recording).astype(np.float32)
            self._recording = []
            return audio

    @property
    def is_recording(self):
        return self._is_recording

    def get_current_buffer(self):
        """Get a copy of the current recording buffer (for real-time preview)."""
        with self._lock:
            if not self._recording:
                return None
            return np.concatenate(self._recording).astype(np.float32)


def normalize_audio(audio):
    """Peak-normalize audio to -1dB headroom."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        target = 10 ** (-1 / 20)  # -1dB ≈ 0.891
        audio = audio * (target / peak)
    return audio


def trim_trailing_silence(audio, sample_rate=16000, threshold=0.01, frame_ms=50, pad_ms=100):
    """Trim silence from the end of audio for better last-syllable recognition.

    Without this, Whisper may reinterpret word endings
    (e.g. 'проаналізуй' → 'проаналізую') due to trailing silence.
    """
    frame_size = int(sample_rate * frame_ms / 1000)
    pad_samples = int(sample_rate * pad_ms / 1000)

    end = len(audio)
    for i in range(len(audio) - frame_size, 0, -frame_size):
        frame = audio[i : i + frame_size]
        rms = np.sqrt(np.mean(frame**2))
        if rms > threshold:
            end = min(i + frame_size + pad_samples, len(audio))
            break

    trimmed_ms = (len(audio) - end) / sample_rate * 1000
    if trimmed_ms > 80:
        logger.debug(f"Trimmed {trimmed_ms:.0f}ms trailing silence")

    return audio[:end]


def compress_internal_silence(
    audio, sample_rate=16000, threshold=0.01, frame_ms=50, max_silence_ms=300, target_silence_ms=200
):
    """Compress long internal pauses for better recognition after pauses.

    When the speaker pauses for 1-3s (thinking), Whisper loses context and
    first words after the pause are poorly recognized. Compressing to target_silence_ms
    gives Whisper more continuous speech.
    """
    frame_size = int(sample_rate * frame_ms / 1000)
    max_silence_frames = int(max_silence_ms / frame_ms)
    target_silence_samples = int(sample_rate * target_silence_ms / 1000)

    num_frames = len(audio) // frame_size
    if num_frames < 2:
        return audio

    frame_rms = np.array(
        [np.sqrt(np.mean(audio[i * frame_size : (i + 1) * frame_size] ** 2)) for i in range(num_frames)]
    )

    is_silent = frame_rms < threshold
    result = []
    i = 0
    total_compressed_ms = 0

    while i < num_frames:
        if not is_silent[i]:
            result.append(audio[i * frame_size : (i + 1) * frame_size])
            i += 1
        else:
            silence_start = i
            while i < num_frames and is_silent[i]:
                i += 1
            silence_len = i - silence_start

            if silence_len > max_silence_frames:
                silent_chunk = audio[silence_start * frame_size : (silence_start + silence_len) * frame_size]
                compressed_ms = silence_len * frame_ms
                total_compressed_ms += compressed_ms - target_silence_ms
                half_target = target_silence_samples // 2
                result.append(silent_chunk[:half_target])
                result.append(silent_chunk[-half_target:])
            else:
                result.append(audio[silence_start * frame_size : i * frame_size])

    remainder = len(audio) - num_frames * frame_size
    if remainder > 0:
        result.append(audio[num_frames * frame_size :])

    if total_compressed_ms > 50:
        logger.debug(f"Compressed {total_compressed_ms:.0f}ms internal silence")

    return np.concatenate(result) if result else audio


def process_audio(audio, sample_rate=16000, use_vad=True):
    """Full audio processing pipeline: VAD trim → normalize → compress silence → trim trailing.

    If use_vad=True and silero-vad is available, uses neural VAD for trimming.
    Falls back to RMS-based trimming otherwise.
    """
    if use_vad:
        audio = vad_trim(audio, sample_rate)
    audio = normalize_audio(audio)
    audio = compress_internal_silence(audio, sample_rate)
    audio = trim_trailing_silence(audio, sample_rate)
    return audio
