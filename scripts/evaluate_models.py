import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from skimage.transform import resize
import matplotlib.pyplot as plt

# Function to evaluate and plot results
def plot_results(input_data, target_data, predicted_data, model_name, sample_index):
    plt.figure(figsize=(15, 5))

    # Plot input data (noisy spectrogram)
    plt.subplot(1, 3, 1)
    plt.title("Input (Noisy)")
    plt.imshow(input_data.squeeze(), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()

    # Plot target data (clean spectrogram)
    plt.subplot(1, 3, 2)
    plt.title("Target (Clean)")
    plt.imshow(target_data.squeeze(), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()

    # Plot predicted data (denoised spectrogram)
    plt.subplot(1, 3, 3)
    plt.title("Prediction (Denoised)")
    plt.imshow(predicted_data.squeeze(), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()

    plt.suptitle(f"Model: {model_name}, Sample: {sample_index}")

    # Save plot
    output_dir = f"../results/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/sample_{sample_index}.png")
    plt.close()

# Function to plot training vs validation loss
def plot_loss(history, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f"Loss for Model: {model_name}")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    output_dir = f"../results/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/loss_curve.png")
    plt.close()

# Function to evaluate all models
def test_all_models(weights_dir, test_dir_clean, test_dir_noisy):
    # Create results directory
    os.makedirs("../results", exist_ok=True)

    # Load all test data
    clean_files = sorted(os.listdir(test_dir_clean))
    noisy_files = sorted(os.listdir(test_dir_noisy))

    if not clean_files or not noisy_files:
        print("Test data directories are empty.")
        return

    X_test = []
    Y_test = []

    for clean_file, noisy_file in zip(clean_files, noisy_files):
        clean_path = os.path.join(test_dir_clean, clean_file)
        noisy_path = os.path.join(test_dir_noisy, noisy_file)

        clean_data = np.load(clean_path)
        noisy_data = np.load(noisy_path)

        clean_data = np.expand_dims(clean_data, axis=-1)
        noisy_data = np.expand_dims(noisy_data, axis=-1)

        X_test.append(noisy_data)
        Y_test.append(clean_data)

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    X_test = (X_test - np.mean(X_test)) / np.std(X_test)
    Y_test = (Y_test - np.mean(Y_test)) / np.std(Y_test)

    print(f"Loaded {len(X_test)} samples for testing.")

    # Load and evaluate each model
    model_files = [f for f in os.listdir(weights_dir) if f.endswith(".keras")]
    if not model_files:
        print("No models found in weights directory.")
        return

    print(f"Models found: {model_files}")

    for model_file in model_files:
        model_path = os.path.join(weights_dir, model_file)
        model_name = os.path.splitext(model_file)[0]

        try:
            print(f"Attempting to load model: {model_path}")
            model = load_model(model_path)
            print(f"Loaded model: {model_name}")

            # Dynamically resize data based on model input shape
            input_shape = model.input_shape[1:]
            print(f"Model {model_name} expects input shape: {input_shape}")

            X_test_resized = np.array([
                resize(sample.squeeze(), input_shape[:2], mode='constant', anti_aliasing=True)
                for sample in X_test
            ])
            X_test_resized = np.expand_dims(X_test_resized, axis=-1)

            Y_test_resized = np.array([
                resize(sample.squeeze(), input_shape[:2], mode='constant', anti_aliasing=True)
                for sample in Y_test
            ])
            Y_test_resized = np.expand_dims(Y_test_resized, axis=-1)

            evaluate_model(model, X_test_resized, Y_test_resized, model_name)

            # Load and plot training history
            history_path = os.path.join(weights_dir, f"{model_name}_history.json")
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history = json.load(f)
                plot_loss(history, model_name)
            else:
                print(f"No training history found for model {model_name}.")

        except Exception as e:
            print(f"Error loading or evaluating model {model_name}: {e}")

# Function to evaluate a single model
def evaluate_model(model, X_test, Y_test, model_name):
    predictions = model.predict(X_test, verbose=1)

    # Calculate loss and metrics
    loss = np.mean((predictions - Y_test) ** 2)
    mae = np.mean(np.abs(predictions - Y_test))
    print(f"Model: {model_name}, Loss: {loss:.4f}, MAE: {mae:.4f}")

    # Generate plots for a few samples
    for i in range(5):  # Generate plots for only 5 samples
        print(f"Saving plot for model {model_name}, sample {i}")
        plot_results(X_test[i], Y_test[i], predictions[i], model_name, i)

if __name__ == "__main__":
    test_all_models(
        weights_dir="../models/weights",
        test_dir_clean="../data/spectrograms/test/clean",
        test_dir_noisy="../data/spectrograms/test/noisy"
    )
