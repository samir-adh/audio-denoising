import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_waveforms(clean_file, noisy_file, sr=8000):
    """
    Plot waveforms for a clean and a noisy audio file.

    Parameters:
    - clean_file: str, path to the clean .wav file.
    - noisy_file: str, path to the noisy .wav file.
    - sr: int, sample rate (default 8000 Hz).
    """
    # Load clean and noisy audio
    clean_audio, _ = librosa.load(clean_file, sr=sr)
    noisy_audio, _ = librosa.load(noisy_file, sr=sr)

    # Plot waveforms
    plt.figure(figsize=(12, 6))

    # Plot clean audio
    plt.subplot(2, 1, 1)
    plt.plot(np.linspace(0, len(clean_audio) / sr, num=len(clean_audio)), clean_audio)
    plt.title('Clean Audio')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot noisy audio
    plt.subplot(2, 1, 2)
    plt.plot(np.linspace(0, len(noisy_audio) / sr, num=len(noisy_audio)), noisy_audio)
    plt.title('Noisy Audio')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

# Example usage
clean_file = "../test/clean/clean_0.wav" 
noisy_file = "../test/noisy/noisy_0.wav"  

plot_waveforms(clean_file, noisy_file)
