import os
import random
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from skimage.transform import resize


def load_random_test_spectrogram(clean_dir, noisy_dir):
    """Load a random test spectrogram from the test directories."""
    clean_files = sorted(os.listdir(clean_dir))
    noisy_files = sorted(os.listdir(noisy_dir))

    if not clean_files or not noisy_files:
        raise FileNotFoundError("Clean or noisy test directories are empty.")

    random_index = random.randint(0, len(clean_files) - 1)
    clean_path = os.path.join(clean_dir, clean_files[random_index])
    noisy_path = os.path.join(noisy_dir, noisy_files[random_index])

    clean_spectrogram = np.load(clean_path)
    noisy_spectrogram = np.load(noisy_path)

    return clean_spectrogram, noisy_spectrogram, clean_files[random_index]




def plot_spectrograms(clean_spectrogram, noisy_spectrogram, cleaned_spectrograms, random_file):
    """Plot clean, noisy, and cleaned spectrograms."""
    plt.figure(figsize=(15, 10))

    # Plot clean spectrogram
    plt.subplot(len(cleaned_spectrograms) + 2, 1, 1)
    plt.title("Clean Spectrogram")
    plt.imshow(clean_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()

    # Plot noisy spectrogram
    plt.subplot(len(cleaned_spectrograms) + 2, 1, 2)
    plt.title("Noisy Spectrogram")
    plt.imshow(noisy_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()

    # Plot cleaned spectrograms for each model
    for i, (model_name, cleaned_spectrogram) in enumerate(cleaned_spectrograms.items(), start=3):
        plt.subplot(len(cleaned_spectrograms) + 2, 1, i)
        plt.title(f"Cleaned Spectrogram ({model_name})")
        plt.imshow(cleaned_spectrogram, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar()

    plt.tight_layout()
    output_dir = "../results/spectrogram_plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{random_file}_spectrogram_plot.png")
    plt.close()


def spectrogram_to_audio(magnitude, phase, hop_length=256):
    """Convert spectrogram back to audio using magnitude and phase."""
    complex_spectrogram = magnitude * np.exp(1j * phase)
    audio = librosa.istft(complex_spectrogram, hop_length=hop_length)
    # Normalize audio to avoid clipping
    audio = audio / np.max(np.abs(audio))
    return audio

def process_spectrograms_with_models(noisy_spectrogram, models_dir):
    """Apply all models to clean the noisy spectrogram."""
    models = [f for f in os.listdir(models_dir) if f.endswith(".keras")]
    if not models:
        raise FileNotFoundError("No models found in the models directory.")

    cleaned_spectrograms = {}

    for model_file in models:
        model_path = os.path.join(models_dir, model_file)
        model_name = os.path.splitext(model_file)[0]

        print(f"Processing spectrogram with model: {model_name}")
        model = load_model(model_path)

        # Dynamically resize spectrogram based on model input shape
        input_shape = model.input_shape[1:3]
        resized_spectrogram = resize(noisy_spectrogram, input_shape, mode='constant', anti_aliasing=True)
        resized_spectrogram = np.expand_dims(resized_spectrogram, axis=(0, -1))  # Add batch and channel dimensions

        # Predict cleaned magnitude
        cleaned_magnitude = model.predict(resized_spectrogram)[0, :, :, 0]

        # Resize back to original shape
        cleaned_magnitude = resize(cleaned_magnitude, noisy_spectrogram.shape, mode='constant', anti_aliasing=True)
        cleaned_spectrograms[model_name] = cleaned_magnitude

    return cleaned_spectrograms

def save_cleaned_audio(cleaned_audio_dict, sample_rate, random_file):
    """Save cleaned audio for each model."""
    output_dir = "../results/cleaned_audio"
    os.makedirs(output_dir, exist_ok=True)

    for model_name, audio_cleaned in cleaned_audio_dict.items():
        # Ensure audio is normalized
        audio_cleaned = audio_cleaned / np.max(np.abs(audio_cleaned))
        output_path = f"{output_dir}/{random_file}_cleaned_{model_name}.wav"
        sf.write(output_path, audio_cleaned, samplerate=sample_rate)
        print(f"Saved cleaned audio for {model_name}: {output_path}")



if __name__ == "__main__":
    # Parameters
    test_clean_dir = "../data/spectrograms/test/clean"
    test_noisy_dir = "../data/spectrograms/test/noisy"
    models_dir = "../models/weights"
    sample_rate = 16000

    # Load a random test spectrogram
    clean_spectrogram, noisy_spectrogram, random_file = load_random_test_spectrogram(test_clean_dir, test_noisy_dir)
    print(f"Loaded random test spectrogram: {random_file}")

    # Assume the phase is from the noisy spectrogram for reconstruction
    magnitude_noisy, phase_noisy = noisy_spectrogram, np.angle(noisy_spectrogram)

    # Apply models to process the noisy spectrogram
    cleaned_spectrograms = process_spectrograms_with_models(magnitude_noisy, models_dir)

    # Convert spectrograms back to audio
    cleaned_audio_dict = {
        model_name: spectrogram_to_audio(cleaned_magnitude, phase_noisy)
        for model_name, cleaned_magnitude in cleaned_spectrograms.items()
    }

    # Plot spectrograms
    plot_spectrograms(clean_spectrogram, magnitude_noisy, cleaned_spectrograms, random_file)

    # Save cleaned audio
    save_cleaned_audio(cleaned_audio_dict, sample_rate, random_file)
