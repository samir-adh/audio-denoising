import os
import numpy as np
import librosa
from scipy.io.wavfile import write
import random
import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def load_esc50_noise(esc50_path, target_classes):
    """
    Load noise samples from the ESC-50 dataset for selected classes.
    """
    metadata = pd.read_csv(os.path.join(esc50_path, 'meta/esc50.csv'))
    noise_files = []
    for target_class in target_classes:
        class_files = metadata[metadata['category'] == target_class]['filename'].tolist()
        noise_files.extend([os.path.join(esc50_path, 'audio', file) for file in class_files])
    return noise_files


def load_librispeech_clean(librispeech_path):
    """
    Load clean speech audio files from the LibriSpeech dataset.
    """
    clean_files = []
    for root, _, files in os.walk(librispeech_path):
        for file in files:
            if file.endswith(('.flac', '.wav')):
                clean_files.append(os.path.join(root, file))
    return clean_files


def audio_to_mel_spectrogram(audio, sr, n_mels=128, n_fft=2048, hop_length=512):
    """
    Convert audio to a normalized Mel spectrogram.
    """
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())  # Normalize



def generate_clean_and_noisy_wav(esc50_path, librispeech_path, output_dir, sr=8000, frame_length=8064):
    """
    Generate two clean and two noisy .wav files from ESC-50 and LibriSpeech datasets.

    Parameters:
    - esc50_path: Path to the ESC-50 dataset.
    - librispeech_path: Path to the LibriSpeech dataset.
    - output_dir: Directory to save the clean and noisy .wav files.
    - sr: Sample rate for audio.
    - frame_length: Fixed length for audio frames.
    """
    # Load noise and clean files
    noise_classes = ['dog', 'rain', 'traffic', 'birds']
    noise_files = load_esc50_noise(esc50_path, noise_classes)
    clean_files = load_librispeech_clean(librispeech_path)

    # Ensure there are enough clean and noise files
    if len(clean_files) < 2 or len(noise_files) < 2:
        raise ValueError("Not enough clean or noise files to generate samples.")

    # Create output directories
    os.makedirs(f"{output_dir}/clean", exist_ok=True)
    os.makedirs(f"{output_dir}/noisy", exist_ok=True)

    for i in range(2):  # Generate 2 clean and 2 noisy files
        # Load a clean file
        clean_file = random.choice(clean_files)
        audio_clean, _ = librosa.load(clean_file, sr=sr)
        audio_clean = librosa.util.fix_length(audio_clean, size=frame_length)

        # Save clean audio as .wav
        clean_path = f"{output_dir}/clean/clean_{i}.wav"
        write(clean_path, sr, (audio_clean * 32767).astype(np.int16))
        print(f"Clean WAV file saved: {clean_path}")

        # Load a noise file
        noise_file = random.choice(noise_files)
        audio_noise, _ = librosa.load(noise_file, sr=sr)
        audio_noise = librosa.util.fix_length(audio_noise, size=frame_length)

        # Blend clean and noise
        noise_level = np.random.uniform(0.2, 0.8)  # Random noise level
        noisy_audio = audio_clean + noise_level * audio_noise
        noisy_audio = np.clip(noisy_audio, -1.0, 1.0)

        # Save noisy audio as .wav
        noisy_path = f"{output_dir}/noisy/noisy_{i}.wav"
        write(noisy_path, sr, (noisy_audio * 32767).astype(np.int16))
        print(f"Noisy WAV file saved: {noisy_path}")


if __name__ == "__main__":
    generate_clean_and_noisy_wav(
        esc50_path='../datasets/ESC-50-master',  # Path to ESC-50 dataset
        librispeech_path='../datasets/LibriSpeech',  # Path to LibriSpeech dataset
        output_dir='../test',  # Directory to save WAV files
        sr=8000,  # Sampling rate
        frame_length=8064  # Frame length
    )
