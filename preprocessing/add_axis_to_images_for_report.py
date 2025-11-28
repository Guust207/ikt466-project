import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import librosa
import librosa.display

#GPT GENERATED SCRIPT for better images to report!

# Paths
path_to_data = Path.cwd() / ".." / "data"
path_to_audio = path_to_data / "audio"
path_to_vis = path_to_data / "visualization_examples"
path_to_vis.mkdir(exist_ok=True)

# Get the first 10 WAV files
wav_files = sorted(list(path_to_audio.glob("*.wav")))[:10]

print(f"Generating visualizations for {len(wav_files)} files...")

for wav_file in wav_files:
    print(f"\nProcessing: {wav_file.name}")
    aud, sr = librosa.load(wav_file)

    # --- Waveform ---
    plt.figure(figsize=(12, 3))
    librosa.display.waveshow(aud, sr=sr)
    plt.title(f"Waveform: {wav_file.name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(path_to_vis / f"{wav_file.stem}_waveform.png", dpi=300)
    plt.close()

    # --- STFT Spectrogram ---
    stft_spec = librosa.stft(aud)
    stft_db = librosa.amplitude_to_db(np.abs(stft_spec), ref=np.max)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(stft_db, sr=sr, x_axis="time", y_axis="log")
    plt.title(f"STFT Spectrogram: {wav_file.name}")
    plt.colorbar(format="%+2.f dB")
    plt.tight_layout()
    plt.savefig(path_to_vis / f"{wav_file.stem}_stft.png", dpi=300)
    plt.close()

    # --- Mel Spectrogram ---
    mel_spec = librosa.feature.melspectrogram(y=aud, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", fmax=8000)
    plt.title(f"Mel Spectrogram: {wav_file.name}")
    plt.colorbar(format="%+2.f dB")
    plt.tight_layout()
    plt.savefig(path_to_vis / f"{wav_file.stem}_mel.png", dpi=300)
    plt.close()

print("\n✨ Completed! All images saved to:", path_to_vis)
