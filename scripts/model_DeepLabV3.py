import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, SpatialDropout2D, Dropout, UpSampling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from skimage.transform import resize
import matplotlib.pyplot as plt
import librosa

# Data Augmentation: Time and Frequency Masking (SpecAugment)
def frequency_masking(spec, freq_width=20):
    freq_start = np.random.randint(0, max(1, spec.shape[0] - freq_width))
    spec[freq_start:freq_start + freq_width, :] = 0
    return spec

def time_masking(spec, time_width=20):
    time_start = np.random.randint(0, max(1, spec.shape[1] - time_width))
    spec[:, time_start:time_start + time_width] = 0
    return spec

# Audio augmentation
def augment_audio(audio, sr, pitch_shift=True, time_stretch=True):
    if pitch_shift:
        n_steps = np.random.uniform(-2, 2)
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    if time_stretch:
        rate = np.random.uniform(0.9, 1.1)
        audio = librosa.effects.time_stretch(audio, rate=rate)
    return audio

# Load and preprocess data
def load_and_preprocess_data(clean_dir, noisy_dir, resize_shape=(128, 32)):
    clean_files = sorted(os.listdir(clean_dir))
    noisy_files = sorted(os.listdir(noisy_dir))
    
    if len(clean_files) != len(noisy_files):
        raise ValueError("Mismatch between clean and noisy files.")

    X_in, X_ou = [], []

    for clean_file, noisy_file in zip(clean_files, noisy_files):
        clean_path = os.path.join(clean_dir, clean_file)
        noisy_path = os.path.join(noisy_dir, noisy_file)

        if not os.path.exists(clean_path) or not os.path.exists(noisy_path):
            raise FileNotFoundError(f"File not found: {clean_path} or {noisy_path}")

        X_ou.append(np.load(clean_path))
        X_in.append(np.load(noisy_path))

    X_in = np.array(X_in)
    X_ou = np.array(X_ou)

    # Resize to match expected shape
    X_in = resize(X_in, (X_in.shape[0], *resize_shape), mode='constant', anti_aliasing=True)
    X_ou = resize(X_ou, (X_ou.shape[0], *resize_shape), mode='constant', anti_aliasing=True)

    # Expand dimensions for channel axis
    X_in = np.expand_dims(X_in, axis=-1)
    X_ou = np.expand_dims(X_ou, axis=-1)

    # Normalize data
    X_in = (X_in - np.mean(X_in)) / (np.std(X_in) + 1e-8)
    X_ou = (X_ou - np.mean(X_ou)) / (np.std(X_ou) + 1e-8)

    return X_in, X_ou

# Learning Rate Scheduler function
def lr_schedule(epoch):
    initial_lr = 1e-4
    if epoch < 10:
        return initial_lr
    else:
        return initial_lr * tf.math.exp(-0.1 * (epoch - 10))

lr_scheduler = LearningRateScheduler(lr_schedule)

# Training function
def train_model(clean_dir, noisy_dir, val_clean_dir, val_noisy_dir, weights_path, epochs, batch_size):
    # Load data
    X_in, X_ou = load_and_preprocess_data(clean_dir, noisy_dir)
    val_X_in, val_X_ou = load_and_preprocess_data(val_clean_dir, val_noisy_dir)

    # Define MobileNetV2 as base model
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(128, 32, 3)
    )

    inp = Input(shape=(128, 32, 1))
    l1 = Conv2D(3, (1, 1))(inp)  
    base_model_output = base_model(l1)

    # Calculate upsampling factors dynamically
    target_height, target_width = 128, 32
    base_output_shape = base_model.output_shape[1:3]
    if None in base_output_shape:
        raise ValueError("Base model output shape could not be determined. Ensure input shape is properly defined.")
    
    upsample_height = target_height // base_output_shape[0]
    upsample_width = target_width // base_output_shape[1]

    x = UpSampling2D(size=(upsample_height, upsample_width))(base_model_output)
    out = Conv2D(1, (1, 1), activation='sigmoid')(x)
    model = Model(inp, out)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='mse',
        metrics=['mae']
    )

    # Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(weights_path, 'best_model.keras'),
        monitor='val_loss',
        save_best_only=True
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Preprocess input for the model
    x_train = X_in  
    x_val = val_X_in

    # Train the model
    base_model.trainable = False 
    history = model.fit(
        x=x_train,
        y=X_ou,
        validation_data=(val_X_in, val_X_ou),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, lr_scheduler, early_stopping]
    )

    # Fine-tuning
    base_model.trainable = True
    history = model.fit(
        x=x_train,
        y=X_ou,
        validation_data=(val_X_in, val_X_ou),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, lr_scheduler, early_stopping]
    )

    # Save the trained model
    model.save(os.path.join(weights_path, 'final_model.h5'))

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    return model

# Main
if __name__ == "__main__":
    train_model(
        clean_dir='../data/spectrograms/train/clean',
        noisy_dir='../data/spectrograms/train/noisy',
        val_clean_dir='../data/spectrograms/val/clean',
        val_noisy_dir='../data/spectrograms/val/noisy',
        weights_path='../models/weights',
        epochs=50,
        batch_size=32  
    )
