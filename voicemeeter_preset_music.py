"""
Voicemeeter Potato preset для PD100X — Спів та Гітара
Оптимізовано для запису/стрімінгу вокалу та акустичної гітари

Різниця від інших пресетів:
- Comp вищий (згладжує динаміку вокалу)
- Gate мінімальний (не ріже тихі ноти/затухання)
- Denoiser мінімальний (зберігає гармоніки)
- Reverb увімкнений (для вокалу)
- Routing: A1 (моніторинг) + B1 (Whisper) + B3 (запис/стрім)

EQ потрібно налаштувати вручну (API обмеження):
  Натисни EQ на Strip 1, потім:
  Band 1: 60Hz  HPF (тип High-Pass) — прибрати гул, зберегти бас
  Band 2: 200Hz +1dB  Q=1.0 — тепло/тіло голосу
  Band 3: 800Hz -1dB  Q=1.0 — прибрати каламутність
  Band 4: 2.5kHz +2dB Q=1.0 — присутність вокалу
  Band 5: 5kHz  +1dB  Q=1.5 — яскравість
  Band 6: 10kHz +1dB  Q=1.0 — повітря/мерехтіння
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
    time.sleep(0.1)


def get_param(name):
    v = ctypes.c_float()
    vm.VBVMR_GetParameterFloat(name.encode(), ctypes.byref(v))
    return v.value


print("=" * 50)
print("Voicemeeter Potato - PD100X Music Preset")
print("=" * 50)

# ── Strip 1 (Hardware Input 1 — PD100X) ──
settings = {
    "Strip[0].A1": 0,       # Monitoring OFF
    "Strip[0].A2": 0,
    "Strip[0].A3": 0,
    "Strip[0].A4": 0,
    "Strip[0].A5": 0,
    "Strip[0].B1": 1,       # Push-to-Talk (Whisper)
    "Strip[0].B2": 0,       # Voice Chat OFF
    "Strip[0].B3": 1,       # Recording/Streaming ON
    "Strip[0].Mono": 1,     # Mono ON
    "Strip[0].Gain": 6,     # Fader +6dB
    "Strip[0].Comp": 5,     # Compressor = 5 (згладжує динаміку)
    "Strip[0].Gate": 1,     # Gate = 1 (мінімальний — не різати тихі ноти)
    "Strip[0].Denoiser": 2, # Denoiser = 2 (мінімальний — зберегти гармоніки)
    "Strip[0].Reverb": 4,   # Reverb = 4 (для вокалу)
    "Strip[0].Delay": 0,    # Delay OFF
    "Strip[0].EQ.on": 1,    # EQ ON (bands set manually)
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
    print("Music preset applied!")
else:
    print("Some settings did not apply. Check Voicemeeter.")
print("\nRouting: B1=Whisper, B3=Recording/Streaming")
print("NOTE: Enable Reverb ON in Master Section for vocal reverb!")
print("\nEQ bands (manual setup required):")
print("  Band 1: 60Hz  HPF")
print("  Band 2: 200Hz +1dB  Q=1.0")
print("  Band 3: 800Hz -1dB  Q=1.0")
print("  Band 4: 2.5kHz +2dB Q=1.0")
print("  Band 5: 5kHz  +1dB  Q=1.5")
print("  Band 6: 10kHz +1dB  Q=1.0")
print("=" * 50)
