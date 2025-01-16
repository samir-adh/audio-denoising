import os
import numpy as np
import matplotlib.pyplot as plt

def inspect_data(clean_dir, noisy_dir, num_samples=5):
    """
    Inspect the clean and noisy datasets to ensure correctness.
    """
    clean_files = sorted(os.listdir(clean_dir))
    noisy_files = sorted(os.listdir(noisy_dir))

    # Verify dataset pairing
    assert len(clean_files) == len(noisy_files), "Mismatch in clean and noisy dataset sizes!"

    for i in range(min(num_samples, len(clean_files))):
        # Load clean and noisy data
        clean_data = np.load(os.path.join(clean_dir, clean_files[i]))
        noisy_data = np.load(os.path.join(noisy_dir, noisy_files[i]))

        # Check shapes
        assert clean_data.shape == noisy_data.shape, f"Shape mismatch: {clean_files[i]} vs {noisy_files[i]}"

        # Print basic statistics
        print(f"Sample {i+1}:")
        print(f"  Clean - Min: {clean_data.min()}, Max: {clean_data.max()}, Mean: {clean_data.mean()}")
        print(f"  Noisy - Min: {noisy_data.min()}, Max: {noisy_data.max()}, Mean: {noisy_data.mean()}")

        # Plot clean and noisy data
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(clean_data, aspect='auto', origin='lower')
        plt.title(f"Clean Spectrogram: {clean_files[i]}")
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(noisy_data, aspect='auto', origin='lower')
        plt.title(f"Noisy Spectrogram: {noisy_files[i]}")
        plt.colorbar()
        plt.show()

# Example usage
inspect_data(
    clean_dir='../data/spectrograms/clean',
    noisy_dir='../data/spectrograms/noisy',
    num_samples=5
)
