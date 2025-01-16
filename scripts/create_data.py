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


def augment_audio(audio, sr, pitch_shift=True, time_stretch=True):
    """
    Apply augmentations to the audio signal.
    """
    if pitch_shift:
        n_steps = np.random.uniform(-2, 2)  
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    if time_stretch:
        rate = np.random.uniform(0.9, 1.1) 
        audio = librosa.effects.time_stretch(audio, rate=rate)
    return audio


def preprocess_and_save(esc50_path, librispeech_path, output_dir, sr, frame_length, hop_length_frame, n_mels, augment=False, test_size=0.2, val_size=0.1):
    """
    Preprocess ESC-50 and LibriSpeech datasets, blend noise with clean speech, split into subsets,
    and save spectrograms for CNN models.
    """
    # Load noise and clean audio files
    noise_classes = ['dog', 'rain', 'traffic', 'birds']
    noise_files = load_esc50_noise(esc50_path, noise_classes)
    clean_files = load_librispeech_clean(librispeech_path)

    # Split the clean files into training, validation, and test subsets
    train_files, test_files = train_test_split(clean_files, test_size=test_size, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=val_size / (1 - test_size), random_state=42)

    # Define subsets
    subsets = {'train': train_files, 'val': val_files, 'test': test_files}

    # Create directories for each subset
    for subset in subsets.keys():
        os.makedirs(f"{output_dir}/{subset}/clean", exist_ok=True)
        os.makedirs(f"{output_dir}/{subset}/noisy", exist_ok=True)

    # Process each subset separately
    for subset, files in subsets.items():
        for file_idx, clean_file in enumerate(tqdm(files, desc=f"Processing {subset.capitalize()} Files")):
            try:
                # Load clean audio
                audio_clean, _ = librosa.load(clean_file, sr=sr)
                audio_clean = librosa.util.fix_length(audio_clean, size=frame_length)

                # Apply augmentation only to training data
                if augment and subset == 'train':
                    audio_clean = augment_audio(audio_clean, sr)
                    audio_clean = librosa.util.fix_length(audio_clean, size=frame_length)

                # Load a random noise file
                noise_file = noise_files[np.random.randint(0, len(noise_files))]
                audio_noise, _ = librosa.load(noise_file, sr=sr)
                audio_noise = librosa.util.fix_length(audio_noise, size=frame_length)

                # Handle length mismatches
                min_length = min(len(audio_clean), len(audio_noise))
                audio_clean = audio_clean[:min_length]
                audio_noise = audio_noise[:min_length]

                # Ensure the final length matches the frame length
                if len(audio_clean) != frame_length or len(audio_noise) != frame_length:
                    raise ValueError(f"Mismatch in adjusted lengths - Clean: {len(audio_clean)}, Noise: {len(audio_noise)}")

                # Blend clean and noise audio
                noise_level = np.random.uniform(0.2, 0.8)  # Random noise level
                noisy_audio = audio_clean + noise_level * audio_noise
                noisy_audio = np.clip(noisy_audio, -1.0, 1.0)  # Ensure values are within the valid range

                # Convert audio to Mel spectrograms
                clean_spec = audio_to_mel_spectrogram(audio_clean, sr, n_mels=n_mels)
                noisy_spec = audio_to_mel_spectrogram(noisy_audio, sr, n_mels=n_mels)

                # Save the spectrograms
                np.save(f"{output_dir}/{subset}/clean/clean_{file_idx}.npy", clean_spec)
                np.save(f"{output_dir}/{subset}/noisy/noisy_{file_idx}.npy", noisy_spec)

            except Exception as e:
                print(f"Error processing file {clean_file}: {e}")


if __name__ == "__main__":
    preprocess_and_save(
        esc50_path='../datasets/ESC-50-master',
        librispeech_path='../datasets/LibriSpeech',
        output_dir='../data/spectrograms',
        sr=8000,  
        frame_length=8064, 
        hop_length_frame=8064,  
        n_mels=128,  
        augment=True,  
        test_size=0.2, 
        val_size=0.1 
    )
