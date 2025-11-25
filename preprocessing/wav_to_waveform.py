import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import librosa
import librosa.display
import librosa.feature


def get_wav_filenames(path_to_dir: Path):
    filenames = [file.name for file in path_to_dir.iterdir() if file.is_file()]
    return filenames


def wav_to_png(_wav_filename: str):
    return _wav_filename.replace(".wav", ".png")

# Paths to data directories
path_to_data = Path.cwd() / ".." / "data"
path_to_audio = path_to_data / "audio"
path_to_waveform = path_to_data / "waveform"
path_to_stft = path_to_data / "STFT"
path_to_mel = path_to_data / "mel"

# Get list of .wav filenames
wav_filenames = get_wav_filenames(path_to_audio)

# List of waveforms relating to the .wav files
waveforms = [] # [[aud1, sr1], [aud2, sr2], ...]

# Use librosa to get all waveforms out of the .wav files
print("Starting .wav --> waveforms...")
for idx, wav_filename in enumerate(wav_filenames):
    wav_file = path_to_audio / wav_filename

    aud, sr = librosa.load(wav_file, sr=22050)
    waveform = aud, sr
    waveforms.append(waveform)

# Find max- and min-amplitude
waveforms_flattened = np.concatenate([aud for aud, _ in waveforms])
min_amp, max_amp = np.min(waveforms_flattened), np.max(waveforms_flattened)

# Plot- and save the waveforms as .png's
no_files = len(waveforms)
print("Starting creating .png's...")
for idx, waveform in enumerate(waveforms):
    # Give filename
    png_filename = wav_to_png(wav_filenames[idx])
    waveform_file = path_to_waveform / png_filename
    stft_file = path_to_stft / png_filename
    mel_file = path_to_mel / png_filename

    # Plot waveforms and save
    plt.plot(waveform[0])
    plt.ylim([min_amp, max_amp])
    plt.axis("off")
    plt.savefig(waveform_file, dpi="figure", bbox_inches="tight", pad_inches=0)
    plt.close()

    # Plot STFT and save
    stft_spec = librosa.stft(waveform[0])
    stft_spec_db = librosa.amplitude_to_db(np.abs(stft_spec), ref=np.max)
    # plt.axis("off")
    librosa.display.specshow(stft_spec_db, x_axis="time", y_axis="hz", fmax=16000)
    #plt.savefig(stft_spec_db, dpi="figure", bbox_inches="tight", pad_inches=0)
    plt.savefig(stft_file)
    plt.close()

    # Plot MEL and save
    mel_spec = librosa.feature.melspectrogram(y=waveform[0], sr=waveform[1])
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # plt.axis("off")
    librosa.display.specshow(mel_spec_db, x_axis="time", y_axis="mel", sr=waveform[1], fmax=8000)
    # plt.savefig(spectrogram_file, dpi="figure", bbox_inches="tight", pad_inches=0)
    plt.savefig(mel_file)
    plt.close()

    print(f"{idx + 1} / {no_files} done.")

    if idx > 1:
        break