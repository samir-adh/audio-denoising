"""
This script preprocesses audio files for the audio denoising project. It performs the following steps:
1. Creates directories to store preprocessed files.
2. Loads clean audio files from the LibriSpeech dataset.
3. Loads noise audio files from the ESC-50 dataset and filters them to select only interior sounds.
4. Resamples noise files to match the sample rate of the clean audio files.
5. Preprocesses each clean audio file by adding noise and saves the resulting noisy audio files.

Functions:
- generate_id: Generates a unique identifier for a given audio signal.
- preprocess_audio_file: Preprocesses a given audio file by adding noise and saves the clean and noisy versions.
- resample: Resamples an audio signal to a specified sample rate.

Constants:
- CLEAN_DEV_DATASET: Path to the clean development dataset.
- CLEAN_TRAIN_DATASET: Path to the clean training dataset.
- MAX_CLEAN_AUDIOS: Maximum number of clean audio files to process.
- NOISE_DATASET: Path to the noise dataset.
- MAX_NOISE_SAMPLES: Maximum number of noise samples to use.

Variables:
- clean_ds_path: Path to the clean dataset.
- clean_files_list: List of paths to clean audio files.
- clean_samplerate: Sample rate of the clean audio files.
- noise_files_list: List of paths to noise audio files.
- sample_noise: Sample noise signal.
- noise_samplerate: Sample rate of the noise audio files.
- seed: Seed for random number generation.
- random_state: Random state for shuffling the file lists.
- n_noises: Number of noise files.
- noises_array: Array of resampled noise signals.
- noises_ids: List of unique identifiers for the noise signals.
- start: Start time for preprocessing.
- stop: Stop time for preprocessing.
"""

import os
import time as Time

import numpy as np
import pandas as pd
import soundfile
from tqdm import tqdm

from helpers import (
    generate_id,
    preprocess_audio_file,
    resample,
)

# Create directories to store preprocessed files
os.makedirs("preprocessed", exist_ok=True)
os.makedirs("preprocessed/clean_audio", exist_ok=True)
os.makedirs("preprocessed/noisy_audio", exist_ok=True)
os.makedirs("preprocessed/temp", exist_ok=True)

# Constants for dataset paths and limits
CLEAN_DEV_DATASET = "datasets/LibriSpeech/dev-clean"
CLEAN_TRAIN_DATASET = "datasets/LibriSpeech/train-clean-100"
MAX_CLEAN_AUDIOS = 256

# Load clean audio files from the dataset
clean_ds_path = CLEAN_DEV_DATASET
clean_files_list = []
for root, dirs, files in os.walk(clean_ds_path):
    for file in files:
        if file.endswith(".flac"):
            fullpath = os.path.join(root, file)
            clean_files_list.append(fullpath)
print(f"There is a total of {len(clean_files_list)} audio files in the dataset.")
_, clean_samplerate = soundfile.read(clean_files_list[0])

# Load noise audio files from the ESC-50 dataset and filter for interior sounds
NOISE_DATASET = "datasets/ESC-50-master/"
audio_dir = os.path.join(NOISE_DATASET, "audio")
labels_path = os.path.join(NOISE_DATASET, "meta/esc50.csv")
labels = pd.read_csv(labels_path)
MAX_NOISE_SAMPLES = 256

# Filter noise dataset to select only interior sounds
interior_noise_labels = labels[labels["target"] // 10 == 3]
noise_files_list = [
    os.path.join(audio_dir, row["filename"])
    for _, row in interior_noise_labels.iterrows()
]
print(f"{len(noise_files_list)} noise files are selected.")
sample_noise, noise_samplerate = soundfile.read(noise_files_list[0])

# Shuffle the clean and noise file lists
seed = 123
random_state = np.random.RandomState(seed)
random_state.shuffle(clean_files_list)
random_state.shuffle(noise_files_list)

# Resample noise files to match the sample rate of clean audio files
n_noises = len(noise_files_list)
noises_array = np.zeros((n_noises, clean_samplerate))
noises_ids = []
for index in range(n_noises):
    noise_file = noise_files_list[index]
    noise_signal, _ = soundfile.read(noise_file)
    cropped_noise = noise_signal[:noise_samplerate]
    resampled_noise = resample(cropped_noise, noise_samplerate, clean_samplerate)
    noises_array[index] = resampled_noise
    noisy_audio_id = generate_id(noise_signal)
    noises_ids.append(noisy_audio_id)

# Limit the number of clean and noise files to process
clean_files_list = clean_files_list[:MAX_CLEAN_AUDIOS]
noises_array = noises_array[:MAX_NOISE_SAMPLES]
noises_ids = noises_ids[:MAX_NOISE_SAMPLES]

# Preprocess each clean audio file by adding noise and save the results
start = Time.time()
for index in tqdm(range(len(clean_files_list))):
    audio_file = clean_files_list[index]
    preprocess_audio_file(
        audio_file=audio_file,
        clean_path="preprocessed/clean_audio",
        noisy_path="preprocessed/noisy_audio",
        clean_samplerate=clean_samplerate,
        noises_array=noises_array,
        noises_ids=noises_ids,
        max_segments=2
    )
stop = Time.time()
print(f"Total time : {stop-start:.3f}s.")
