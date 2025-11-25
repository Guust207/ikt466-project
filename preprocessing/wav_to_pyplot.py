import matplotlib.pyplot as plt
import librosa.display
from pathlib import Path



def get_wav_filenames(path_to_dir: Path):
    filenames = [file.name for file in path_to_dir.iterdir() if file.is_file()]
    return filenames


def wav_to_png(wav_filename: str):
    return wav_filename.replace(".wav", ".png")


# Paths to directories
path_to_data = Path.cwd() / ".." / "data"
path_to_audio = path_to_data / "audio"
path_to_pyplot = path_to_data / "pyplot"

wav_filenames = get_wav_filenames(path_to_audio)

spectrograms = []  # [y1, sr1], [y2, sr2], ...]
max_amp = 0
min_amp = 0

# Loop through all wav files and use librosa for spectrograms
print("Starting audio --> spectrograms...")
for idx, audio_filename in enumerate(wav_filenames):
    audio_file = Path(path_to_audio / audio_filename)
    spectrogram = librosa.load(audio_file)
    spectrograms.append(spectrogram)

    max_y = max(spectrogram[0])
    min_y = min(spectrogram[0])

    max_amp = max_y if max_y > max_amp else max_amp
    min_amp = min_y if min_y < min_amp else min_amp

# Loop through all spectrograms and plot them
print("Starting creating .png's")
no_files = len(spectrograms)
for idx, spectrogram in enumerate(spectrograms):
    # Plot
    plt.plot(spectrogram[0])
    plt.ylim([min_amp, max_amp])
    plt.axis("off")
    plt.draw()

    # Proper filename
    png_filename = wav_to_png(wav_filenames[idx])
    file_to_save = path_to_pyplot / png_filename

    # Save file
    plt.savefig(file_to_save, dpi="figure", bbox_inches="tight", pad_inches=0)
    plt.close()

    # Print status
    print(f"{idx + 1} / {no_files} done")
