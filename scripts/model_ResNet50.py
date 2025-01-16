import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, SpatialDropout2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
import segmentation_models as sm
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import librosa

# Data Augmentation
def frequency_masking(spec, freq_width=20):
    freq_start = np.random.randint(0, spec.shape[0] - freq_width)
    spec[freq_start:freq_start+freq_width, :] = 0
    return spec

def time_masking(spec, time_width=20):
    time_start = np.random.randint(0, spec.shape[1] - time_width)
    spec[:, time_start:time_start+time_width] = 0
    return spec

def augment_audio(audio, sr, pitch_shift=True, time_stretch=True):
    if pitch_shift:
        n_steps = np.random.uniform(-2, 2)
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    if time_stretch:
        rate = np.random.uniform(0.9, 1.1)
        audio = librosa.effects.time_stretch(audio, rate=rate)
    return audio

def mixup(x1, x2, alpha=0.2):
    lambda_val = np.random.beta(alpha, alpha)
    return lambda_val * x1 + (1 - lambda_val) * x2

# Load and preprocess data
def load_and_preprocess_data(clean_dir, noisy_dir, resize_shape=(128, 32)):
    clean_files = sorted(os.listdir(clean_dir))
    noisy_files = sorted(os.listdir(noisy_dir))
    
    X_in = [] 
    X_ou = []  

    for clean_file, noisy_file in zip(clean_files, noisy_files):
        clean_path = os.path.join(clean_dir, clean_file)
        noisy_path = os.path.join(noisy_dir, noisy_file)

        X_ou.append(np.load(clean_path))
        X_in.append(np.load(noisy_path))

    X_in = np.array(X_in)
    X_ou = np.array(X_ou)

    # Resize spectrograms to expected shape
    X_in = resize(X_in, (X_in.shape[0], *resize_shape), mode='constant', anti_aliasing=True)
    X_ou = resize(X_ou, (X_ou.shape[0], *resize_shape), mode='constant', anti_aliasing=True)

    # Expand dimensions to add channel axis
    X_in = np.expand_dims(X_in, axis=-1)
    X_ou = np.expand_dims(X_ou, axis=-1)

    # Normalize data
    X_in = (X_in - np.mean(X_in)) / np.std(X_in)
    X_ou = (X_ou - np.mean(X_ou)) / np.std(X_ou)

    return X_in, X_ou

# Train the model
def train_model(clean_dir, noisy_dir, val_clean_dir, val_noisy_dir, weights_path, epochs, batch_size):
    # Load and preprocess the training and validation data
    X_in, X_ou = load_and_preprocess_data(clean_dir, noisy_dir)
    val_X_in, val_X_ou = load_and_preprocess_data(val_clean_dir, val_noisy_dir)

    # U-Net model with ResNet50 backbone
    base_model = sm.Unet('resnet50', encoder_weights='imagenet')  
    inp = Input(shape=(128, 32, 1)) 
    l1 = Conv2D(3, (1, 1))(inp)  
    l1 = SpatialDropout2D(0.3)(l1) 
    out = base_model(l1)  
    model = Model(inp, out)

    # Compile model with Adam optimizer and learning rate scheduler
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='mse',
        metrics=['mae']
    )

    # Callbacks for better training
    checkpoint = ModelCheckpoint(
        os.path.join(weights_path, 'model_ResNet50_best.keras'), 
        monitor='val_loss', 
        save_best_only=True, 
        mode='auto'
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.1, 
        patience=5, 
        min_lr=1e-6
    )

    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        X_in, X_ou,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_X_in, val_X_ou),
        callbacks=[checkpoint, lr_scheduler, early_stopping]
    )

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    return model

# Example: Training the model with separate validation data
if __name__ == "__main__":
    train_model(
        clean_dir='../data/spectrograms/train/clean', 
        noisy_dir='../data/spectrograms/train/noisy',
        val_clean_dir='../data/spectrograms/val/clean',  
        val_noisy_dir='../data/spectrograms/val/noisy',  
        weights_path='../models/weights', 
        epochs=50, 
        batch_size=16
    )
