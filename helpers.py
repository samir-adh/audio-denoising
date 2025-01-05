"""
This module provides helper functions for audio processing, including plotting signals,
adding noise, resampling, and storing audio data.

Functions:
    plot_signal(signal, samplerate):
        Plots a single audio signal.

    plot_2signals(signal1, signal2, samplerate):
        Plots two audio signals for comparison.

    plotSpectrogram(spectrum, samplerate, hop_length):
        Plots the spectrogram of an audio signal.

    resample(signal, source_samplerate, target_samplerate):
        Resamples an audio signal to a different sample rate.

    add_noise(signal: np.ndarray, noise: np.ndarray, noise_level=0.01):
        Adds noise to an audio signal.

    add_all_noises(segment: np.ndarray, noises: np.ndarray, noise_level=0.01):
        Adds multiple noise signals to an audio segment.

    store_stft(signal, filepath, n_fft, hop_length):
        Computes and stores the Short-Time Fourier Transform (STFT) of an audio signal.

    store_audio(signal, filepath, samplerate):
        Stores an audio signal to a file.

    generate_id(object: np.ndarray) -> str:
        Generates a unique ID for a given numpy array.

    chop_audio(signal, samplerate):
        Chops an audio signal into segments of one second each.

        Preprocesses an audio segment by adding noise and storing both clean and noisy versions.

    preprocess_audio_file(
        Preprocesses an entire audio file by chopping it into segments and adding noise.
"""

import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy
import soundfile


def plot_signal(signal, samplerate):
    """
    Plots two audio signals for comparison.

    Parameters:
    signal1 (np.ndarray): First audio signal.
    signal2 (np.ndarray): Second audio signal.
    samplerate (int): Sample rate of the audio signals.

    Returns:
    None
    """
    fs = 1 / samplerate
    duration = signal.shape[0] * fs
    time = np.arange(stop=duration, step=fs)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.plot(time, signal)


def plot_2signals(signal1, signal2, samplerate):
    """
    Plots the spectrogram of an audio signal.

    Parameters:
    spectrum (np.ndarray): Spectrogram of the audio signal.
    samplerate (int): Sample rate of the audio signal.
    hop_length (int): Number of audio samples between adjacent STFT columns.

    Returns:
    None
    """
    fs = 1 / samplerate
    duration = signal1.shape[0] * fs
    time = np.arange(stop=duration, step=fs)
    plt.figure(figsize=(16, 9))
    # Signal1
    plt.subplot(2, 1, 1)
    plt.plot(time, signal1)
    plt.title("Signal 1")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Signal2
    plt.subplot(2, 1, 2)
    plt.plot(time, signal2)
    plt.title("Signal 2")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


def plotSpectrogram(spectrum, samplerate, hop_length):
    """
    Plots the spectrogram of an audio signal.

    Parameters:
    spectrum (np.ndarray): Spectrogram of the audio signal.
    samplerate (int): Sample rate of the audio signal.
    hop_length (int): Number of audio samples between adjacent STFT columns.

    Returns:
    None
    """
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


def resample(signal, source_samplerate, target_samplerate):
    """
    Resamples an audio signal to a different sample rate.

    Parameters:
    signal (np.ndarray): Audio signal to be resampled.
    source_samplerate (int): Original sample rate of the audio signal.
    target_samplerate (int): Target sample rate for the audio signal.

    Returns:
    np.ndarray: Resampled audio signal.
    """
    resampled_signal = scipy.signal.resample_poly(
        signal, target_samplerate, source_samplerate
    )
    return resampled_signal


def add_noise(signal: np.ndarray, noise: np.ndarray, noise_level=0.01):
    """
    Adds noise to an audio signal.

    Parameters:
    signal (np.ndarray): Original audio signal.
    noise (np.ndarray): Noise signal to be added.
    noise_level (float, optional): Level of noise to be added. Default is 0.01.

    Returns:
    np.ndarray: Noisy audio signal.
    """
    if len(signal) != len(noise):
        raise Exception(f"Arguments `signal` and `noise` must have the same dimensions \
                        \n signal  shape : {signal.shape}, noise signal shape : {noise.shape}")
    rd_noise_level = (
        1 + 0.2 * np.random.random()
    ) * noise_level  # to randomize the amplitude of the noise
    noisy_audio = signal + rd_noise_level * noise
    return noisy_audio


def add_all_noises(segment: np.ndarray, noises: np.ndarray, noise_level=0.01):
    """
    Adds multiple noise signals to an audio segment.

    Parameters:
    segment (np.ndarray): Original audio segment.
    noises (np.ndarray): Array of noise signals to be added.
    noise_level (float, optional): Level of noise to be added. Default is 0.01.

    Returns:
    np.ndarray: Noisy audio segment.
    """
    n_noises = noises.shape[0]
    input = np.full((n_noises, segment.shape[0]), segment)
    random_amplitude = (
        (1 + (np.random.random((n_noises)) - 0.5) * 0.6) * noise_level
    )  # the noise level will be randomly increased/decreased by 30% of the specified noise level
    random_noise = noises * random_amplitude[:, np.newaxis]
    output = input + random_noise
    return output


def store_stft(signal, filepath, n_fft, hop_length):
    """
    Computes and stores the Short-Time Fourier Transform (STFT) of an audio signal.

    Parameters:
    signal (np.ndarray): Audio signal.
    filepath (str): Path to save the STFT.
    n_fft (int): Number of FFT components.
    hop_length (int): Number of audio samples between adjacent STFT columns.

    Returns:
    None
    """
    stft_signal = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    stft_signal.shape
    magnitude = np.abs(stft_signal)
    if os.path.exists(filepath):
        raise Exception(f"Hash collision for file : {filepath}.")
    np.save(filepath, magnitude)
    return filepath


def store_audio(signal, filepath, samplerate):
    """
    Stores an audio signal to a file.

    Parameters:
    signal (np.ndarray): Audio signal to be stored.
    filepath (str): Path to save the audio file.
    samplerate (int): Sample rate of the audio signal.

    Returns:
    None
    """
    soundfile.write(filepath, signal, samplerate)


def generate_id(object: np.ndarray) -> str:
    """
    Generates a unique ID for a given numpy array.

    Parameters:
    object (np.ndarray): Numpy array for which to generate the ID.

    Returns:
    str: Unique ID for the numpy array.
    """
    return str(np.abs(hash(object.tobytes())))


def chop_audio(signal, samplerate):
    """
    Chops an audio signal into segments of one second each.

    Parameters:
    signal (np.ndarray): Audio signal to be chopped.
    samplerate (int): Sample rate of the audio signal.

    Returns:
    list[np.ndarray]: List of audio segments.
    """
    n_segments = len(signal) // samplerate
    segments = np.zeros((n_segments, samplerate))
    for index in range(n_segments):
        start = index * samplerate
        stop = (index + 1) * samplerate
        segments[index] = signal[start:stop]
    return segments


def preprocess_segment(
    segment: np.ndarray,
    noises_array: np.ndarray,
    noises_ids: list[str],
    clean_path: str,
    noisy_path: str,
    clean_samplerate: int,
):
    """
    Preprocesses an audio segment by adding noise and storing both clean and noisy versions.

    Parameters:
    segment (np.ndarray): Audio segment to be processed.
    noises_array (np.ndarray): Array of noise signals to be added.
    noises_ids (list[str]): List of noise identifiers corresponding to the noises_array.
    clean_path (str): Path to save the clean audio segment.
    noisy_path (str): Path to save the noisy audio segment.
    clean_samplerate (int): Sample rate of the clean audio.

    Returns:
    None
    """
    segment_id = generate_id(segment)
    segment_filename = segment_id + ".flac"
    segment_fullpath = os.path.join(clean_path, segment_filename)
    store_audio(segment, segment_fullpath, clean_samplerate)
    noisy_audios = add_all_noises(segment, noises_array)
    for index in range(noisy_audios.shape[0]):
        noisy_audio = noisy_audios[index]
        noisy_audio_id = noises_ids[index]
        noisy_audio_filename = segment_id + "_" + noisy_audio_id + ".flac"
        noisy_audio_fullpath = os.path.join(noisy_path, noisy_audio_filename)
        store_audio(noisy_audio, noisy_audio_fullpath, clean_samplerate)


def preprocess_audio_file(
    audio_file: str,
    clean_path: str,
    noisy_path: str,
    clean_samplerate: int,
    noises_array: np.ndarray,
    noises_ids: list[str],
    n_fft=512,
    hop_length=None,
    fcount=[0],
    max_segments=4,
):
    """
    Preprocess an audio file by chopping it into segments and applying noise to each segment.

    Parameters:
    audio_file (str): Path to the input audio file.
    clean_path (str): Path to save the clean audio segments.
    noisy_path (str): Path to save the noisy audio segments.
    clean_samplerate (int): Sample rate of the clean audio.
    noises_array (np.ndarray): Array of noise signals to be added to the audio segments.
    noises_ids (list[str]): List of noise identifiers corresponding to the noises_array.
    n_fft (int, optional): Number of FFT components. Default is 512.
    hop_length (int, optional): Number of audio samples between adjacent STFT columns. Default is n_fft // 8.
    fcount (list[int], optional): Counter for the number of processed files. Default is [0].
    max_segments (int, optional): Maximum number of segments to process from the audio file. Default is 4.

    Returns:
    None
    """
    if hop_length is None:
        hop_length = n_fft // 8
    signal, _ = soundfile.read(audio_file)
    segments = chop_audio(signal, clean_samplerate)
    n_segments = min(len(segments), max_segments)
    for index in range(n_segments):
        segment = segments[index]
        preprocess_segment(
            segment,
            noises_array,
            noises_ids,
            clean_path,
            noisy_path,
            clean_samplerate,
        )
