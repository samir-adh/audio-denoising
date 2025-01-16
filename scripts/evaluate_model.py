import os
import numpy as np
from skimage.transform import resize
from keras.models import load_model
import matplotlib.pyplot as plt

def load_test_data(clean_dir, noisy_dir):
    clean_files = sorted(os.listdir(clean_dir))
    noisy_files = sorted(os.listdir(noisy_dir))

    assert len(clean_files) == len(noisy_files), "Mismatch between clean and noisy files"

    X_test = []  
    Y_test = []  

    for clean_file, noisy_file in zip(clean_files, noisy_files):
        clean_path = os.path.join(clean_dir, clean_file)
        noisy_path = os.path.join(noisy_dir, noisy_file)

        Y_test.append(np.load(clean_path))
        X_test.append(np.load(noisy_path))

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    # Resize to model input dimensions
    X_test = resize(X_test, (X_test.shape[0], 128, 32), mode='constant', anti_aliasing=True)
    Y_test = resize(Y_test, (Y_test.shape[0], 128, 32), mode='constant', anti_aliasing=True)

    # Expand dimensions to match input shape (samples, height, width, channels)
    X_test = np.expand_dims(X_test, axis=-1)
    Y_test = np.expand_dims(Y_test, axis=-1)

    # Normalize data
    X_test = (X_test - np.mean(X_test)) / np.std(X_test)
    Y_test = (Y_test - np.mean(Y_test)) / np.std(Y_test)

    # Handle NaN or infinite values
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    Y_test = np.nan_to_num(Y_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Check if NaN or infinite values exist after processing
    print("NaN or Inf check for X_test:", np.any(np.isnan(X_test)), np.any(np.isinf(X_test)))
    print("NaN or Inf check for Y_test:", np.any(np.isnan(Y_test)), np.any(np.isinf(Y_test)))

    return X_test, Y_test

clean_dir = '../data/spectrograms/clean'
noisy_dir = '../data/spectrograms/noisy'

X_test, Y_test = load_test_data(clean_dir, noisy_dir)

model = load_model('../models/weights/model_ResNet.keras')

test_loss, test_mae = model.evaluate(X_test, Y_test, batch_size=16)
print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

predictions = model.predict(X_test[:5])

for i in range(5):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(Y_test[i].squeeze(), aspect='auto', origin='lower', cmap='viridis')
    plt.title('Ground Truth Clean Spectrogram')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(X_test[i].squeeze(), aspect='auto', origin='lower', cmap='viridis')
    plt.title('Noisy Spectrogram')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(predictions[i].squeeze(), aspect='auto', origin='lower', cmap='viridis')
    plt.title('Predicted Clean Spectrogram')
    plt.colorbar()

    plt.tight_layout()
    plt.show()
