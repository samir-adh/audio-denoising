import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
from skimage.transform import resize
print(tf.__version__)

# Unet network
def unet(pretrained_weights=None, input_size=(128, 32, 1)):
    size_filter_in = 16
    kernel_init = 'he_normal'
    activation_layer = None
    inputs = Input(input_size)
    conv1 = Conv2D(size_filter_in, 3, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(inputs)
    conv1 = LeakyReLU()(conv1)
    conv1 = Conv2D(size_filter_in, 3, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(conv1)
    conv1 = LeakyReLU()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(size_filter_in * 2, 3, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(pool1)
    conv2 = LeakyReLU()(conv2)
    conv2 = Conv2D(size_filter_in * 2, 3, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(conv2)
    conv2 = LeakyReLU()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(size_filter_in * 4, 3, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(pool2)
    conv3 = LeakyReLU()(conv3)
    conv3 = Conv2D(size_filter_in * 4, 3, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(conv3)
    conv3 = LeakyReLU()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(size_filter_in * 8, 3, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(pool3)
    conv4 = LeakyReLU()(conv4)
    conv4 = Conv2D(size_filter_in * 8, 3, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(conv4)
    conv4 = LeakyReLU()(conv4)
    drop4 = Dropout(0.6)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(size_filter_in * 16, 3, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(pool4)
    conv5 = LeakyReLU()(conv5)
    conv5 = Conv2D(size_filter_in * 16, 3, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(conv5)
    conv5 = LeakyReLU()(conv5)
    drop5 = Dropout(0.6)(conv5)

    up6 = Conv2D(size_filter_in * 8, 2, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(UpSampling2D(size=(2, 2))(drop5))
    up6 = LeakyReLU()(up6)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(size_filter_in * 8, 3, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(merge6)
    conv6 = LeakyReLU()(conv6)
    conv6 = Conv2D(size_filter_in * 8, 3, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(conv6)
    conv6 = LeakyReLU()(conv6)
    up7 = Conv2D(size_filter_in * 4, 2, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(UpSampling2D(size=(2, 2))(conv6))
    up7 = LeakyReLU()(up7)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(size_filter_in * 4, 3, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(merge7)
    conv7 = LeakyReLU()(conv7)
    conv7 = Conv2D(size_filter_in * 4, 3, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(conv7)
    conv7 = LeakyReLU()(conv7)
    up8 = Conv2D(size_filter_in * 2, 2, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(UpSampling2D(size=(2, 2))(conv7))
    up8 = LeakyReLU()(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(size_filter_in * 2, 3, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(merge8)
    conv8 = LeakyReLU()(conv8)
    conv8 = Conv2D(size_filter_in * 2, 3, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(conv8)
    conv8 = LeakyReLU()(conv8)

    up9 = Conv2D(size_filter_in, 2, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(UpSampling2D(size=(2, 2))(conv8))
    up9 = LeakyReLU()(up9)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(size_filter_in, 3, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(merge9)
    conv9 = LeakyReLU()(conv9)
    conv9 = Conv2D(size_filter_in, 3, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(conv9)
    conv9 = LeakyReLU()(conv9)
    conv9 = Conv2D(2, 3, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(conv9)
    conv9 = LeakyReLU()(conv9)
    conv10 = Conv2D(1, 1, activation='tanh')(conv9)

    model = Model(inputs, conv10)

    model.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=['mae'])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

def training(clean_dir, noisy_dir, val_clean_dir, val_noisy_dir, weights_path, name_model, training_from_scratch, epochs, batch_size):
    """ Train the U-Net model using spectrogram data from clean and noisy directories """

    # Load training data
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

    # Load validation data
    val_clean_files = sorted(os.listdir(val_clean_dir))
    val_noisy_files = sorted(os.listdir(val_noisy_dir))
    assert len(val_clean_files) == len(val_noisy_files), "Mismatch between validation clean and noisy files"

    X_val_in = []
    X_val_ou = []

    for val_clean_file, val_noisy_file in zip(val_clean_files, val_noisy_files):
        val_clean_path = os.path.join(val_clean_dir, val_clean_file)
        val_noisy_path = os.path.join(val_noisy_dir, val_noisy_file)

        X_val_ou.append(np.load(val_clean_path))
        X_val_in.append(np.load(val_noisy_path))

    X_val_in = np.array(X_val_in)
    X_val_ou = np.array(X_val_ou)

    # Resize spectrograms to (128, 32) if not in the expected shape
    X_in = resize(X_in, (X_in.shape[0], 128, 32), mode='constant', anti_aliasing=True)
    X_ou = resize(X_ou, (X_ou.shape[0], 128, 32), mode='constant', anti_aliasing=True)
    X_val_in = resize(X_val_in, (X_val_in.shape[0], 128, 32), mode='constant', anti_aliasing=True)
    X_val_ou = resize(X_val_ou, (X_val_ou.shape[0], 128, 32), mode='constant', anti_aliasing=True)

    # Expand dimensions to add the channel axis
    X_in = np.expand_dims(X_in, axis=-1)
    X_ou = np.expand_dims(X_ou, axis=-1)
    X_val_in = np.expand_dims(X_val_in, axis=-1)
    X_val_ou = np.expand_dims(X_val_ou, axis=-1)

    # Normalize data
    X_in = (X_in - np.mean(X_in)) / np.std(X_in)
    X_ou = (X_ou - np.mean(X_ou)) / np.std(X_ou)
    X_val_in = (X_val_in - np.mean(X_val_in)) / np.std(X_val_in)
    X_val_ou = (X_val_ou - np.mean(X_val_ou)) / np.std(X_val_ou)

    # Create U-Net model
    if training_from_scratch:
        model = unet(input_size=(128, 32, 1))
    else:
        model = unet(pretrained_weights=os.path.join(weights_path, name_model + '.keras'), input_size=(128, 32, 1))

    # Save best model weights
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(weights_path, 'unet_model.keras'),
        monitor='val_loss',
        save_best_only=True,
        mode='auto',
        verbose=1
    )

    # Learning rate reduction
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    # Train model with validation data
    history = model.fit(
        X_in, X_ou,
        validation_data=(X_val_in, X_val_ou),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[checkpoint, reduce_lr, early_stopping],
        verbose=1
    )

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    training(
        clean_dir='../data/spectrograms/train/clean',
        noisy_dir='../data/spectrograms/train/noisy',
        val_clean_dir='../data/spectrograms/val/clean',
        val_noisy_dir='../data/spectrograms/val/noisy',
        weights_path='../models/weights',
        name_model='unet_model',
        training_from_scratch=True,
        epochs=60,
        batch_size=8
    )
