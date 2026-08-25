import os
import wave
import struct
import numpy as np

# Path Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SOUNDS_DIR = os.path.join(PROJECT_ROOT, "sounds")
os.makedirs(SOUNDS_DIR, exist_ok=True)


def create_beep(filename, freq=440, duration=0.3, volume=0.5, pulses=1):
    """Generates a clean sine wave .wav sound file."""
    sample_rate = 44100
    n_samples = int(sample_rate * duration)
    filepath = os.path.join(SOUNDS_DIR, filename)

    audio = []
    pulse_length = n_samples // pulses

    for i in range(n_samples):
        # Create silent gaps between pulses if multiple
        current_pulse = (i // (pulse_length // 2)) % 2
        if pulses > 1 and current_pulse == 1:
            val = 0.0
        else:
            val = volume * np.sin(2 * np.pi * freq * (i / sample_rate))
        
        # Convert float (-1.0 to 1.0) to 16-bit PCM integer
        packed_sample = struct.pack('<h', int(val * 32767))
        audio.append(packed_sample)

    with wave.open(filepath, 'w') as wav_file:
        wav_file.setnchannels(1)        # Mono sound
        wav_file.setsampwidth(2)        # 2 bytes = 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b''.join(audio))

    print(f"🔊 Created sound: {filename}")


def main():
    print("🎵 Generating alert sound files...")
    # Level 1: Soft Low Beep (440 Hz - Pitch A4)
    create_beep("alert_low.wav", freq=440, duration=0.4, volume=0.4, pulses=1)

    # Level 2: Medium Warning (660 Hz - Pitch E5, 2 pulses)
    create_beep("alert_medium.wav", freq=660, duration=0.5, volume=0.6, pulses=2)

    # Level 3: Loud Emergency Siren (880 Hz - Pitch A5, 4 rapid pulses)
    create_beep("alert_high.wav", freq=880, duration=0.6, volume=0.8, pulses=4)

    print("✅ All sound files generated in /sounds directory!")


if __name__ == "__main__":
    main()