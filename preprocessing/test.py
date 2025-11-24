import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
from pathlib import Path
import numpy as np


# TODO:
# Retrieve .wav files
# Convert to Mel
# Create Mel Spectrogram

path_to_wav = Path.cwd() / ".." / "data" / "audio"

# arr = ["1-11687-A-47.wav", "1-24796-A-47.wav", "1-18631-A-23.wav"]
arr = [x for x in path_to_wav.glob("**/*") if x.is_file()]
print(arr[0].exists())