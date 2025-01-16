import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

# Directories
CLEAN_AUDIO_DIR = "preprocessed/clean_audio"
NOISY_AUDIO_DIR = "preprocessed/noisy_audio"
CLEAN_SPEC_DIR = "preprocessed/clean_spectrograms"
NOISY_SPEC_DIR = "preprocessed/noisy_spectrograms"

os.makedirs(CLEAN_SPEC_DIR, exist_ok=True)
os.makedirs(NOISY_SPEC_DIR, exist_ok=True)

# Parameters
SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# Helper Functions
def generate_spectrogram(audio_path, target_shape=(128, 128)):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    resized_spectrogram = librosa.util.fix_length(spectrogram_db, size=target_shape[1], axis=1)
    return resized_spectrogram[:target_shape[0], :]


def convert_audio_to_spectrogram(audio_dir, output_dir, target_shape=(128, 128)):
    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")]
    for audio_file in audio_files:
        try:
            spec = generate_spectrogram(audio_file, target_shape=target_shape)
            basename = os.path.basename(audio_file).replace(".wav", ".npy")
            np.save(os.path.join(output_dir, basename), spec)
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")


# Build Model
def build_denoising_autoencoder(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    output_layer = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="mse")
    return model

def train_model():
    # Convert audio to spectrograms
    print("Converting clean audio to spectrograms...")
    convert_audio_to_spectrogram(CLEAN_AUDIO_DIR, CLEAN_SPEC_DIR)
    print("Converting noisy audio to spectrograms...")
    convert_audio_to_spectrogram(NOISY_AUDIO_DIR, NOISY_SPEC_DIR)

    # Load preprocessed data
    clean_files = sorted([f for f in os.listdir(CLEAN_SPEC_DIR) if f.endswith(".npy")])
    noisy_files = sorted([f for f in os.listdir(NOISY_SPEC_DIR) if f.endswith(".npy")])

    # Print loaded files for debugging
    print(f"Clean files: {clean_files[:5]}... ({len(clean_files)} total)")
    print(f"Noisy files: {noisy_files[:5]}... ({len(noisy_files)} total)")

    # Match noisy files with corresponding clean files
    matched_clean_files = []
    matched_noisy_files = []

    for clean_file in clean_files:
        base_name = clean_file.replace("_clean.npy", "")
        noisy_matches = [f for f in noisy_files if base_name in f]
        if noisy_matches:
            matched_clean_files.extend([clean_file] * len(noisy_matches))
            matched_noisy_files.extend(noisy_matches)

    print(f"Matched clean files: {len(matched_clean_files)}")
    print(f"Matched noisy files: {len(matched_noisy_files)}")

    if len(matched_clean_files) == 0 or len(matched_noisy_files) == 0:
        raise ValueError("No matching spectrogram files found.")

    if len(matched_clean_files) == 0 or len(matched_noisy_files) == 0:
        print("No matches found. Debugging filenames:")
        print(f"Clean files: {clean_files}")
        print(f"Noisy files: {noisy_files}")
        raise ValueError("No matching spectrogram files found.")

    # Continue with loading and training...


    # Load spectrograms
    X_clean = np.array([np.load(os.path.join(CLEAN_SPEC_DIR, f)) for f in matched_clean_files])
    X_noisy = np.array([np.load(os.path.join(NOISY_SPEC_DIR, f)) for f in matched_noisy_files])

    # Reshape for CNN input
    X_clean = np.expand_dims(X_clean, axis=-1)
    X_noisy = np.expand_dims(X_noisy, axis=-1)

    # Train-test split
    split_idx = int(0.8 * len(X_clean))
    X_train_clean, X_val_clean = X_clean[:split_idx], X_clean[split_idx:]
    X_train_noisy, X_val_noisy = X_noisy[:split_idx], X_noisy[split_idx:]

    # Build model
    input_shape = X_train_noisy.shape[1:]  # (height, width, channels)
    model = build_denoising_autoencoder(input_shape)

    # Train model
    model.fit(
        X_train_noisy,
        X_train_clean,
        validation_data=(X_val_noisy, X_val_clean),
        epochs=50,
        batch_size=16
    )

    # Save the model
    model.save("denoising_autoencoder.h5")
    print("Model training complete. Saved as 'denoising_autoencoder.h5'.")

if __name__ == "__main__":
    train_model()
