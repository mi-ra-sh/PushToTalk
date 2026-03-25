"""
Voicemeeter Potato preset для PD100X — Gaming Voice Chat
Оптимізовано для Discord/TeamSpeak/in-game voice chat

Різниця від Whisper-пресету:
- Gate вищий (відсікає клавіатуру/мишу між фразами)
- Comp вищий (вирівнює крик vs шепіт)
- Denoiser агресивніший
- Routing: A1 (моніторинг) + B2 (voice chat app)
  B1 залишається для Push-to-Talk/Whisper

EQ потрібно налаштувати вручну (API обмеження):
  Натисни EQ на Strip 1, потім:
  Band 1: 100Hz HPF (тип High-Pass) — прибрати гул + удари по столу
  Band 2: 250Hz -3dB  Q=1.0 — менше бубніння
  Band 3: 1.5kHz +2dB Q=1.0 — чіткість голосу
  Band 4: 3kHz   +4dB Q=1.5 — розбірливість (головна для voice chat)
  Band 5: 6kHz   +2dB Q=1.5 — яскравість
  Band 6: 12kHz  -3dB Q=1.0 — прибрати сичання + шум навушників
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
print("Voicemeeter Potato - PD100X Gaming Preset")
print("=" * 50)

# ── Strip 1 (Hardware Input 1 — PD100X) ──
settings = {
    "Strip[0].A1": 0,       # Monitoring OFF
    "Strip[0].A2": 0,
    "Strip[0].A3": 0,
    "Strip[0].A4": 0,
    "Strip[0].A5": 0,
    "Strip[0].B1": 1,       # Push-to-Talk (Whisper) — залишаємо
    "Strip[0].B2": 1,       # Voice Chat ON (Discord/TeamSpeak)
    "Strip[0].B3": 0,
    "Strip[0].Mono": 1,     # Mono ON
    "Strip[0].Gain": 6,     # Fader +6dB
    "Strip[0].Comp": 4,     # Compressor = 4 (сильніший — крик/шепіт)
    "Strip[0].Gate": 2,     # Gate = 2 (клавіатура/миша)
    "Strip[0].Denoiser": 6, # Denoiser = 6 (агресивніший)
    "Strip[0].Reverb": 0,   # Reverb OFF
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
    print("Gaming preset applied!")
else:
    print("Some settings did not apply. Check Voicemeeter.")
print("\nRouting: B1=Whisper, B2=Voice Chat")
print("In Discord: set input to 'Voicemeeter Out B2'")
print("\nEQ bands (manual setup required):")
print("  Band 1: 100Hz HPF")
print("  Band 2: 250Hz -3dB  Q=1.0")
print("  Band 3: 1.5kHz +2dB Q=1.0")
print("  Band 4: 3kHz  +4dB  Q=1.5")
print("  Band 5: 6kHz  +2dB  Q=1.5")
print("  Band 6: 12kHz -3dB  Q=1.0")
print("=" * 50)
