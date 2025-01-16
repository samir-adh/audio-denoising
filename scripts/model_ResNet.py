import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

from tensorflow import keras
import segmentation_models as sm
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Conv2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam  
import matplotlib.pyplot as plt
from skimage.transform import resize

def train_model(clean_dir, noisy_dir, weights_path, epochs, batch_size):
    # Load all clean and noisy spectrograms
    clean_files = sorted(os.listdir(clean_dir))
    noisy_files = sorted(os.listdir(noisy_dir))
    assert len(clean_files) == len(noisy_files), "Mismatch between clean and noisy files"

    X_in = [] 
    X_ou = []  

    for clean_file, noisy_file in zip(clean_files, noisy_files):
        clean_path = os.path.join(clean_dir, clean_file)
        noisy_path = os.path.join(noisy_dir, noisy_file)

        X_ou.append(np.load(clean_path))
        X_in.append(np.load(noisy_path))

    X_in = np.array(X_in)
    X_ou = np.array(X_ou)

    # Resize spectrograms to (128, 32) if they are not already in the expected shape
    X_in = resize(X_in, (X_in.shape[0], 128, 32), mode='constant', anti_aliasing=True)
    X_ou = resize(X_ou, (X_ou.shape[0], 128, 32), mode='constant', anti_aliasing=True)

    # Expand dimensions to add the channel axis
    X_in = np.expand_dims(X_in, axis=-1)
    X_ou = np.expand_dims(X_ou, axis=-1)

    # Normalize data
    X_in = (X_in - np.mean(X_in)) / np.std(X_in)  
    X_ou = (X_ou - np.mean(X_ou)) / np.std(X_ou)

    # Train-validation split
    x_train, x_val, y_train, y_val = train_test_split(X_in, X_ou, test_size=0.10, random_state=42)

    # Create U-Net model
    base_model = sm.Unet('resnet34', encoder_weights=None) 
    inp = Input(shape=(128, 32, 1))  
    l1 = Conv2D(3, (1, 1))(inp) 
    out = base_model(l1)
    model = Model(inp, out)

    # Compile model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-4),  
        loss='mse',
        metrics=['mae']
    )

    # Create directory for weights
    os.makedirs(weights_path, exist_ok=True)

    # Save best model weights
    checkpoint = ModelCheckpoint(
        os.path.join(weights_path, 'model_ResNet.keras'),
        monitor='val_loss',
        save_best_only=True,
        mode='auto'
    )

    # Train model
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[checkpoint]
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

if __name__ == "__main__":
    train_model(
        clean_dir='../data/spectrograms/clean',
        noisy_dir='../data/spectrograms/noisy',
        weights_path='../models/weights',
        epochs=20,
        batch_size=16
    )
