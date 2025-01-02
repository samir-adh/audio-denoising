import numpy as np
import librosa
import scipy
import matplotlib.pyplot as plt
import scipy.signal


def plotSpectrogram(spectrum, samplerate, hop_length):
    magnitude = np.abs(spectrum)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(
        librosa.amplitude_to_db(magnitude, ref=np.max),
        sr=samplerate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="log",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("STFT Magnitude Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()
