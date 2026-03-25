"""
Voicemeeter Potato preset для PD100X — оптимізація для Push-to-Talk / Whisper
Застосовує налаштування на Strip 1 (Hardware Input 1) через Voicemeeter Remote API

Принцип: менше обробки = краще для Whisper (тренувався на 680k годин реального аудіо).
Агресивна обробка створює артефакти в мел-спектрограмі.

Параметричний EQ потрібно налаштувати вручну (API обмеження):
  Натисни EQ на Strip 1, потім:
  Band 1: 80Hz  HPF (тип High-Pass)
  Band 2: 200Hz  -2dB  Q=1.0
  Band 3: 1kHz   +2dB  Q=1.0
  Band 4: 3kHz   +4dB  Q=1.5
  Band 5: 6kHz   +2dB  Q=1.5
  Band 6: 14kHz  -2dB  Q=1.0
"""
import ctypes
import time
import sys

VM_DLL = r"C:\Program Files (x86)\VB\Voicemeeter\VoicemeeterRemote64.dll"

try:
    vm = ctypes.cdll.LoadLibrary(VM_DLL)
except OSError:
    print("Voicemeeter Remote DLL not found!")
    sys.exit(1)

ret = vm.VBVMR_Login()
if ret != 0:
    print(f"Login failed (code {ret}). Is Voicemeeter Potato running?")
    sys.exit(1)

time.sleep(0.2)


def set_param(name, val):
    vm.VBVMR_SetParameters(f"{name}={val};".encode())
    time.sleep(0.3)


def get_param(name):
    v = ctypes.c_float()
    vm.VBVMR_GetParameterFloat(name.encode(), ctypes.byref(v))
    return v.value


print("=" * 50)
print("Voicemeeter Potato - PD100X Preset")
print("=" * 50)

# ── Strip 1 (Hardware Input 1 — PD100X) ──
settings = {
    "Strip[0].A1": 0,       # Monitoring OFF
    "Strip[0].A2": 0,
    "Strip[0].A3": 0,
    "Strip[0].A4": 0,
    "Strip[0].A5": 0,
    "Strip[0].B1": 1,       # Push-to-Talk ON
    "Strip[0].B2": 0,
    "Strip[0].B3": 0,
    "Strip[0].Mono": 1,     # Mono ON (Whisper працює з моно)
    "Strip[0].Gain": -2,    # Fader -2dB (було -5; аналіз показав RMS -34..-39dB — занизько)
    "Strip[0].Comp": 2.5,   # Compressor = 2.5 (вирівнює динамічний діапазон, Whisper не scale-invariant)
    "Strip[0].Gate": 2.5,   # Gate = 2.5 (відсікає фон між словами)
    "Strip[0].Denoiser": 3, # Denoiser = 3 (знижено з 4 — менше спектральних артефактів для Whisper)
    "Strip[0].Limit": -3,   # Limiter = -3dB (запобігає кліпінгу на голосних звуках)
    "Strip[0].EQGain1": -6, # Low EQ: -6dB (було -3; бас 56-67% домінує, proximity effect)
    "Strip[0].EQGain2": 3,  # Mid EQ: +3dB (було +1.5; мід 1-9% — занадто мало для розбірливості)
    "Strip[0].EQGain3": 2.5,# High EQ: +2.5dB (було +1; верхи 2-10% — приголосні нечіткі)
    "Strip[0].Reverb": 0,   # Reverb OFF
    "Strip[0].Delay": 0,    # Delay OFF
    "Strip[0].EQ.on": 1,    # Parametric EQ ON (bands set manually)
    "Strip[0].Color_x": 0,  # fx echo OFF
    "Strip[0].Color_y": 0,  # brightness reset
}

for name, val in settings.items():
    set_param(name, val)

time.sleep(0.3)
vm.VBVMR_IsParametersDirty()
time.sleep(0.2)

# Verify
print("\n--- Verify ---")
all_ok = True
for name, expected in settings.items():
    actual = get_param(name)
    label = name.split(".", 1)[1]
    ok = abs(actual - expected) < 0.01
    if not ok:
        all_ok = False
    status = "OK" if ok else "MISMATCH"
    print(f"  {label:12s} = {actual:5.1f}  ({status})")

vm.VBVMR_Logout()

print("\n" + "=" * 50)
if all_ok:
    print("All settings applied successfully!")
else:
    print("Some settings did not apply. Check Voicemeeter.")
print("\nParametric EQ bands (manual setup required):")
print("  Band 1: 80Hz  HPF")
print("  Band 2: 200Hz  -2dB  Q=1.0")
print("  Band 3: 1kHz   +2dB  Q=1.0")
print("  Band 4: 3kHz   +4dB  Q=1.5")
print("  Band 5: 6kHz   +2dB  Q=1.5")
print("  Band 6: 14kHz  -2dB  Q=1.0")
print("\nKey changes vs previous preset (2026-02-26 audio analysis):")
print("  Gain:     -5 → -2   (RMS був -34..-39dB — занизько)")
print("  EQGain1:  -3 → -6   (бас 56-67% домінував)")
print("  EQGain2: +1.5 → +3  (мід 1-9% — занадто мало)")
print("  EQGain3:  +1 → +2.5 (верхи 2-10% — приголосні нечіткі)")
print("=" * 50)
