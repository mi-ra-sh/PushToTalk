"""
PushToTalk Audio Analysis Tool
Record control phrases, analyze audio quality, compare processing steps, generate reports.

Usage:
    python analyze_audio.py              # Record + full analysis
    python analyze_audio.py --quick      # Only 3 phrases
    python analyze_audio.py --no-whisper # Only audio metrics (no model)
    python analyze_audio.py --dir <path> # Analyze existing WAV files
"""

import os
import sys
import time
import argparse
import wave
import logging

import numpy as np
import sounddevice as sd
from scipy import signal

# Fix Windows console encoding
if sys.platform == "win32":
    os.system("chcp 65001 >nul 2>&1")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "audio_analysis")

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("audio_analysis")

# Control phrases for testing
CONTROL_PHRASES = [
    {"id": "short_uk", "text": "Зроби", "desc": "Short imperative (1 word)"},
    {"id": "medium_uk", "text": "Відкрий файл і подивись", "desc": "Medium sentence (4 words)"},
    {"id": "long_uk", "text": "Проаналізуй цей код і зроби commit на remote", "desc": "Long mixed UK+EN"},
    {"id": "shch", "text": "Щоденно перевіряй щось", "desc": "Щ sounds (hard for Whisper)"},
    {"id": "borshch", "text": "Борщ з часником і хлібом", "desc": "Щ ending + typical UA words"},
    {"id": "imperative", "text": "Закрий термінал і запусти deploy", "desc": "Imperatives + tech term"},
    {"id": "english", "text": "Make a commit and push to remote", "desc": "Full English sentence"},
    {"id": "mixed", "text": "Запусти Docker container і перевір Kubernetes", "desc": "Mixed UK+EN tech"},
    {"id": "numbers", "text": "Двадцять п'ять відсотків на двісті рядків", "desc": "Numbers in Ukrainian"},
    {"id": "pause", "text": "Зроби... ось це... і ще оте", "desc": "Sentence with pauses"},
]

QUICK_PHRASES = [CONTROL_PHRASES[0], CONTROL_PHRASES[2], CONTROL_PHRASES[3]]


# === Audio Recording ===

def find_device(device_name="Voicemeeter Out B1"):
    """Find audio input device by name."""
    for i, d in enumerate(sd.query_devices()):
        if device_name in d["name"] and d["max_input_channels"] > 0:
            if "WASAPI" in d["name"] or "Voicemeeter VAIO" in d["name"]:
                return i
    # Fallback: any matching device
    for i, d in enumerate(sd.query_devices()):
        if device_name in d["name"] and d["max_input_channels"] > 0:
            return i
    return None


def record_phrase(phrase_info, device_index=None, sample_rate=16000):
    """Record a single phrase. Returns numpy array or None."""
    print(f"\n  Phrase: \"{phrase_info['text']}\"")
    print(f"  ({phrase_info['desc']})")
    input("  Press ENTER to start recording, then speak the phrase...")

    print("  >> RECORDING... (press ENTER to stop)")

    recording = []
    is_recording = True

    def callback(indata, frames, time_info, status):
        if is_recording:
            recording.append(indata.copy().flatten())

    stream = sd.InputStream(
        callback=callback, channels=1, samplerate=sample_rate,
        blocksize=4096, dtype="float32", device=device_index,
    )
    stream.start()
    input()  # Wait for ENTER
    is_recording = False
    stream.stop()
    stream.close()

    if not recording:
        print("  [!] No audio recorded")
        return None

    audio = np.concatenate(recording).astype(np.float32)
    duration = len(audio) / sample_rate
    rms = np.sqrt(np.mean(audio ** 2))
    print(f"  Recorded: {duration:.1f}s, RMS={rms:.4f}")
    return audio


def save_wav(audio, path, sample_rate=16000):
    """Save audio as 16-bit WAV."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio_norm = audio / peak * 0.95
    else:
        audio_norm = audio
    audio_int16 = (audio_norm * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def load_wav(path):
    """Load WAV file as float32 numpy array. Returns (audio, sample_rate)."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


# === Audio Metrics ===

def compute_metrics(audio, sample_rate=16000):
    """Compute comprehensive audio metrics."""
    if len(audio) == 0:
        return {}

    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    duration = len(audio) / sample_rate

    # Dynamic range (peak to RMS ratio in dB)
    if rms > 0:
        dynamic_range_db = 20 * np.log10(peak / rms) if peak > 0 else 0
        rms_db = 20 * np.log10(rms)
    else:
        dynamic_range_db = 0
        rms_db = -100

    peak_db = 20 * np.log10(peak) if peak > 0 else -100

    # SNR estimate: signal = top 10% RMS frames, noise = bottom 10%
    frame_size = int(sample_rate * 0.02)  # 20ms frames
    n_frames = len(audio) // frame_size
    if n_frames > 2:
        frame_rms = np.array([
            np.sqrt(np.mean(audio[i * frame_size:(i + 1) * frame_size] ** 2))
            for i in range(n_frames)
        ])
        sorted_rms = np.sort(frame_rms)
        n10 = max(1, n_frames // 10)
        noise_rms = np.mean(sorted_rms[:n10])
        signal_rms = np.mean(sorted_rms[-n10:])
        snr_db = 20 * np.log10(signal_rms / noise_rms) if noise_rms > 0 else 60
    else:
        snr_db = 0
        noise_rms = 0

    # Silence ratio (frames with RMS < 0.005)
    if n_frames > 0:
        silence_threshold = 0.005
        silence_frames = np.sum(frame_rms < silence_threshold) if n_frames > 2 else 0
        silence_ratio = silence_frames / n_frames
    else:
        silence_ratio = 0

    # Spectral analysis: energy distribution across frequency bands
    if len(audio) >= 512:
        freqs, psd = signal.welch(audio, fs=sample_rate, nperseg=min(1024, len(audio)))
        total_energy = np.sum(psd)
        if total_energy > 0:
            # Frequency bands
            bands = {
                "sub_bass_0_100": (0, 100),
                "bass_100_300": (100, 300),
                "low_mid_300_1k": (300, 1000),
                "mid_1k_3k": (1000, 3000),
                "high_mid_3k_6k": (3000, 6000),
                "high_6k_8k": (6000, 8000),
            }
            band_energy = {}
            for name, (lo, hi) in bands.items():
                mask = (freqs >= lo) & (freqs < hi)
                band_energy[name] = np.sum(psd[mask]) / total_energy * 100
        else:
            band_energy = {}
    else:
        band_energy = {}

    return {
        "duration_s": round(duration, 2),
        "rms": round(rms, 5),
        "rms_db": round(rms_db, 1),
        "peak": round(peak, 5),
        "peak_db": round(peak_db, 1),
        "dynamic_range_db": round(dynamic_range_db, 1),
        "snr_db": round(snr_db, 1),
        "silence_ratio": round(silence_ratio, 3),
        "band_energy": {k: round(v, 1) for k, v in band_energy.items()},
    }


def format_metrics(metrics, label=""):
    """Format metrics as readable text."""
    lines = []
    if label:
        lines.append(f"  [{label}]")
    lines.append(f"    Duration: {metrics['duration_s']}s")
    lines.append(f"    RMS: {metrics['rms']:.5f} ({metrics['rms_db']:.1f} dB)")
    lines.append(f"    Peak: {metrics['peak']:.5f} ({metrics['peak_db']:.1f} dB)")
    lines.append(f"    Dynamic range: {metrics['dynamic_range_db']:.1f} dB")
    lines.append(f"    SNR: {metrics['snr_db']:.1f} dB")
    lines.append(f"    Silence ratio: {metrics['silence_ratio']:.1%}")
    if metrics.get("band_energy"):
        lines.append("    Spectral balance:")
        band_names = {
            "sub_bass_0_100": "<100Hz",
            "bass_100_300": "100-300",
            "low_mid_300_1k": "300-1k",
            "mid_1k_3k": "1k-3k",
            "high_mid_3k_6k": "3k-6k",
            "high_6k_8k": "6k-8k",
        }
        for band, pct in metrics["band_energy"].items():
            name = band_names.get(band, band)
            bar = "#" * int(pct / 2)
            lines.append(f"      {name:>10}: {pct:5.1f}% {bar}")
    return "\n".join(lines)


# === Visualization ===

def plot_waveform_spectrogram(audio, sample_rate, title, output_path):
    """Generate waveform + spectrogram plot and save as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        fig.suptitle(title, fontsize=12, fontweight="bold")

        # Waveform
        time_axis = np.arange(len(audio)) / sample_rate
        ax1.plot(time_axis, audio, linewidth=0.5, color="#2196F3")
        ax1.set_ylabel("Amplitude")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylim(-1, 1)
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Waveform")

        # Spectrogram
        if len(audio) >= 256:
            ax2.specgram(audio, Fs=sample_rate, NFFT=512, noverlap=256, cmap="viridis")
            ax2.set_ylabel("Frequency (Hz)")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylim(0, min(8000, sample_rate / 2))
            ax2.set_title("Spectrogram")
        else:
            ax2.text(0.5, 0.5, "Audio too short for spectrogram", ha="center", va="center")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return True
    except Exception as e:
        logger.warning(f"Plot error: {e}")
        return False


def plot_comparison(stages, sample_rate, title, output_path):
    """Plot waveforms of all processing stages side by side."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(stages)
        fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n))
        fig.suptitle(title, fontsize=12, fontweight="bold")

        if n == 1:
            axes = [axes]

        for ax, (label, audio) in zip(axes, stages):
            time_axis = np.arange(len(audio)) / sample_rate
            rms = np.sqrt(np.mean(audio ** 2))
            ax.plot(time_axis, audio, linewidth=0.5, color="#2196F3")
            ax.set_ylabel("Amp")
            ax.set_ylim(-1, 1)
            ax.set_title(f"{label}  (dur={len(audio)/sample_rate:.2f}s, RMS={rms:.4f})", fontsize=10)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return True
    except Exception as e:
        logger.warning(f"Comparison plot error: {e}")
        return False


# === Processing Steps ===

def get_processing_stages(audio, sample_rate=16000):
    """Apply processing steps incrementally and return each stage.

    Returns list of (label, audio_array) tuples.
    """
    from audio_engine import vad_trim, normalize_audio, compress_internal_silence, trim_trailing_silence

    stages = [("raw", audio.copy())]

    # Stage 1: VAD trim
    after_vad = vad_trim(audio.copy(), sample_rate)
    stages.append(("+VAD", after_vad))

    # Stage 2: + normalize
    after_norm = normalize_audio(after_vad.copy())
    stages.append(("+normalize", after_norm))

    # Stage 3: + compress silence
    after_compress = compress_internal_silence(after_norm.copy(), sample_rate)
    stages.append(("+compress", after_compress))

    # Stage 4: + trim trailing
    after_trim = trim_trailing_silence(after_compress.copy(), sample_rate)
    stages.append(("+trim_trail", after_trim))

    return stages


# === Whisper Transcription ===

def transcribe_audio(engine, audio, language, sample_rate=16000):
    """Transcribe audio using WhisperEngine. Returns (text, time_ms)."""
    t_start = time.perf_counter()
    result = engine.transcribe(audio, language, sample_rate)
    elapsed_ms = (time.perf_counter() - t_start) * 1000
    text = result.text if result else "[no result]"
    return text, elapsed_ms


# === Recommendations ===

def generate_recommendations(all_metrics):
    """Generate recommendations based on audio metrics."""
    recs = []

    # Average across all raw recordings
    raw_metrics = [m for label, m in all_metrics if label == "raw"]
    if not raw_metrics:
        return ["No raw audio data to analyze"]

    avg_rms = np.mean([m["rms"] for m in raw_metrics])
    avg_snr = np.mean([m["snr_db"] for m in raw_metrics])
    avg_silence = np.mean([m["silence_ratio"] for m in raw_metrics])

    # Level check
    if avg_rms < 0.01:
        recs.append("[!] LEVEL TOO LOW: Average RMS={:.4f}. Increase Voicemeeter Gain or move closer to mic.".format(avg_rms))
    elif avg_rms < 0.03:
        recs.append("[~] Level slightly low: RMS={:.4f}. Consider increasing Gain by 3-5dB.".format(avg_rms))
    elif avg_rms > 0.3:
        recs.append("[!] LEVEL TOO HIGH: RMS={:.4f}. Reduce Gain to avoid clipping.".format(avg_rms))
    else:
        recs.append("[OK] Input level good: RMS={:.4f}".format(avg_rms))

    # SNR check
    if avg_snr < 15:
        recs.append("[!] SNR LOW ({:.0f}dB): Increase Voicemeeter Gate (currently 2.5, try 3-4) or Denoiser.".format(avg_snr))
    elif avg_snr < 25:
        recs.append("[~] SNR moderate ({:.0f}dB): Consider increasing Gate slightly.".format(avg_snr))
    else:
        recs.append("[OK] SNR good: {:.0f}dB".format(avg_snr))

    # Silence ratio
    if avg_silence > 0.5:
        recs.append("[~] High silence ratio ({:.0%}): VAD trimming helps. Consider shorter recordings.".format(avg_silence))

    # Spectral balance
    band_data = [m["band_energy"] for m in raw_metrics if m.get("band_energy")]
    if band_data:
        avg_bass = np.mean([b.get("bass_100_300", 0) for b in band_data])
        avg_mid = np.mean([b.get("mid_1k_3k", 0) for b in band_data])
        avg_high = np.mean([b.get("high_mid_3k_6k", 0) for b in band_data])

        if avg_bass > 40:
            recs.append("[~] Excessive bass ({:.0f}%): Lower Voicemeeter EQGain1 (bass).".format(avg_bass))
        if avg_mid < 15:
            recs.append("[~] Weak midrange ({:.0f}%): Speech clarity may suffer. Increase EQGain2.".format(avg_mid))
        if avg_high < 5:
            recs.append("[~] Weak highs ({:.0f}%): Consonants may be unclear. Increase EQGain3.".format(avg_high))

    # Processing pipeline analysis
    processed_metrics = [m for label, m in all_metrics if label == "+trim_trail"]
    if processed_metrics and raw_metrics:
        avg_raw_dur = np.mean([m["duration_s"] for m in raw_metrics])
        avg_proc_dur = np.mean([m["duration_s"] for m in processed_metrics])
        trim_pct = (1 - avg_proc_dur / avg_raw_dur) * 100 if avg_raw_dur > 0 else 0

        if trim_pct > 50:
            recs.append(f"[~] Processing removes {trim_pct:.0f}% of audio — check if VAD is too aggressive.")
        elif trim_pct > 20:
            recs.append(f"[OK] Processing trims {trim_pct:.0f}% of audio (normal for PTT).")
        else:
            recs.append(f"[OK] Minimal trimming ({trim_pct:.0f}%) — audio is clean.")

    return recs


# === Main Analysis ===

def run_analysis(phrases, device_name="Voicemeeter Out B1", use_whisper=True,
                 from_dir=None, sample_rate=16000):
    """Run full audio analysis pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw_dir = os.path.join(OUTPUT_DIR, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("  PUSHTOTALK AUDIO ANALYSIS REPORT")
    report_lines.append(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 60)
    report_lines.append("")

    # Load Whisper engine if needed
    engine = None
    if use_whisper:
        print("\nLoading Whisper model...")
        from config import load_config
        from whisper_engine import WhisperEngine
        config = load_config()
        engine = WhisperEngine(config)
        print(f"Model loaded: {engine.model_info}")
        report_lines.append(f"Model: {engine.model_info}")
        report_lines.append("")

    # Collect audio
    recordings = {}
    if from_dir:
        # Load existing WAV files
        print(f"\nLoading WAV files from {from_dir}...")
        for f in sorted(os.listdir(from_dir)):
            if f.endswith(".wav"):
                path = os.path.join(from_dir, f)
                audio, sr = load_wav(path)
                if sr != sample_rate:
                    print(f"  [!] {f}: sample rate {sr} != {sample_rate}, skipping")
                    continue
                name = os.path.splitext(f)[0]
                recordings[name] = {"audio": audio, "desc": f, "text": ""}
                print(f"  Loaded: {f} ({len(audio)/sr:.1f}s)")
    else:
        # Record phrases
        device_index = find_device(device_name)
        if device_index is not None:
            dev_info = sd.query_devices(device_index)
            print(f"\nUsing device: {dev_info['name']}")
            report_lines.append(f"Device: {dev_info['name']}")
        else:
            print(f"\n[!] Device '{device_name}' not found, using default")
            report_lines.append("Device: (default)")
        report_lines.append("")

        print(f"\nRecording {len(phrases)} control phrases...")
        print("For each phrase, press ENTER to start, speak, then press ENTER to stop.\n")

        for phrase in phrases:
            audio = record_phrase(phrase, device_index, sample_rate)
            if audio is not None:
                recordings[phrase["id"]] = {
                    "audio": audio,
                    "desc": phrase["desc"],
                    "text": phrase["text"],
                }
                # Save raw WAV
                wav_path = os.path.join(raw_dir, f"{phrase['id']}.wav")
                save_wav(audio, wav_path, sample_rate)
                print(f"  Saved: {wav_path}")

    if not recordings:
        print("\n[!] No recordings to analyze")
        return

    # Analyze each recording
    all_metrics = []
    print(f"\n{'='*60}")
    print("  ANALYSIS")
    print(f"{'='*60}")

    for name, rec in recordings.items():
        audio = rec["audio"]
        desc = rec["desc"]
        expected = rec.get("text", "")

        print(f"\n--- {name}: {desc} ---")
        report_lines.append(f"\n{'='*60}")
        report_lines.append(f"  {name}: {desc}")
        if expected:
            report_lines.append(f"  Expected: \"{expected}\"")
        report_lines.append(f"{'='*60}")

        # Get processing stages
        stages = get_processing_stages(audio, sample_rate)

        # Metrics for each stage
        for label, stage_audio in stages:
            metrics = compute_metrics(stage_audio, sample_rate)
            all_metrics.append((label, metrics))
            report_lines.append(format_metrics(metrics, label))
            if label == "raw":
                print(format_metrics(metrics, label))

        # Visualizations
        # Waveform + spectrogram for raw audio
        plot_path = os.path.join(OUTPUT_DIR, f"{name}_waveform.png")
        if plot_waveform_spectrogram(audio, sample_rate, f"{name}: {desc}", plot_path):
            print(f"  Plot: {plot_path}")

        # Comparison of all stages
        comp_path = os.path.join(OUTPUT_DIR, f"{name}_stages.png")
        if plot_comparison(stages, sample_rate, f"{name}: Processing Stages", comp_path):
            print(f"  Stages plot: {comp_path}")

        # Whisper transcription for each stage
        if engine:
            language = "uk"
            if name == "english" or (expected and all(ord(c) < 128 or c == ' ' for c in expected)):
                language = "en"

            report_lines.append("\n  Whisper transcriptions:")
            print("\n  Whisper transcriptions:")

            for label, stage_audio in stages:
                text, ms = transcribe_audio(engine, stage_audio, language, sample_rate)
                line = f"    {label:>12}: \"{text}\"  ({ms:.0f}ms)"
                report_lines.append(line)
                print(line)

            if expected:
                report_lines.append(f"    {'expected':>12}: \"{expected}\"")
                print(f"    {'expected':>12}: \"{expected}\"")

    # Recommendations
    print(f"\n{'='*60}")
    print("  RECOMMENDATIONS")
    print(f"{'='*60}")
    report_lines.append(f"\n\n{'='*60}")
    report_lines.append("  RECOMMENDATIONS")
    report_lines.append(f"{'='*60}")

    recs = generate_recommendations(all_metrics)
    for rec in recs:
        print(f"  {rec}")
        report_lines.append(f"  {rec}")

    # Save report
    report_path = os.path.join(OUTPUT_DIR, "audio_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Report saved: {report_path}")
    print(f"  Output dir: {OUTPUT_DIR}")

    # Cleanup Whisper
    if engine:
        engine.cleanup()


def main():
    parser = argparse.ArgumentParser(description="PushToTalk Audio Analysis Tool")
    parser.add_argument("--quick", action="store_true", help="Only 3 control phrases")
    parser.add_argument("--no-whisper", action="store_true", help="Skip Whisper transcription")
    parser.add_argument("--dir", type=str, help="Analyze existing WAV files from directory")
    parser.add_argument("--device", type=str, default="Voicemeeter Out B1", help="Audio input device name")
    args = parser.parse_args()

    phrases = QUICK_PHRASES if args.quick else CONTROL_PHRASES

    print("=" * 60)
    print("  PUSHTOTALK AUDIO ANALYSIS TOOL")
    print("=" * 60)

    if args.dir:
        print(f"\n  Mode: Analyze existing files from {args.dir}")
    elif args.quick:
        print(f"\n  Mode: Quick ({len(phrases)} phrases)")
    else:
        print(f"\n  Mode: Full ({len(phrases)} phrases)")

    if args.no_whisper:
        print("  Whisper: DISABLED (metrics only)")
    else:
        print("  Whisper: ENABLED (will transcribe each stage)")

    run_analysis(
        phrases=phrases,
        device_name=args.device,
        use_whisper=not args.no_whisper,
        from_dir=args.dir,
    )


if __name__ == "__main__":
    main()
