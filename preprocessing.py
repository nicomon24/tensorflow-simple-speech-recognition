'''
    Preprocess data and create MFCC or log_mel_spectrograms
'''

import numpy as np
import librosa
import random

class Preprocessor:

    def __init__(self, feature_count=40, window_size_ms=20, window_stride_ms=10):
        self.feature_count = feature_count
        self.window_size_ms = window_size_ms
        self.window_stride_ms = window_stride_ms
        # Setting properties
        self.sample_rate = 16000
        self.sample_length = 16000
        self.window_size_samples = int(self.window_size_ms * self.sample_rate / 1000)
        self.window_stride_samples = int(self.window_stride_ms * self.sample_rate / 1000)
        # Compute derived properties
        self.window_number = 1 + int(self.sample_length / self.window_stride_samples)
        self.fingerprint_size = self.window_number * self.feature_count

    # Clip file if needed
    def check_audio_length(self, wave_audio):
        if len(wave_audio) > self.sample_length:
            # Cut to fixed length
            wave_audio = wave_audio[:self.sample_length]
        elif len(wave_audio) < self.sample_length:
            # Pad with zeros at the end
            delta = self.sample_length - len(wave_audio)
            wave_audio = np.pad(wave_audio, (0,delta), 'constant', constant_values=0)
        return wave_audio

    def add_noise(self, data, noises, k=0.3):
        # Choose a random noise
        noise = random.choice(noises)
        # Add it to the sound
        noise_energy = np.sqrt(np.sum(noise ** 2))
        data_energy = np.sqrt(np.sum(data ** 2))
        if data_energy > 0:
            result = data + (k * noise * data_energy / noise_energy).astype(np.int16)
        else:
            result = k * noise * 0.01 # Reduce more
        return result

    def time_shift(self, data, shift=0):
        result = data
        if shift > 0:
            # First check that the first part has low volume
            if np.sum(data[:shift]**2) < 0.01 * np.sum(data**2):
                # Pad and slice
                result = np.pad(data, (0, shift), 'constant', constant_values=0)[shift:]
        else:
            # Check if the last part has low volume
            if np.sum(data[shift:]**2) < 0.01 * np.sum(data**2):
                # Pad and slice
                result = np.pad(data, (-shift, 0), 'constant', constant_values=0)[:shift]
        return result

    def get_log_mel_spectrograms(self, wave_audio):
        spectrogram = librosa.feature.melspectrogram(wave_audio, sr=self.sample_rate,
                    n_mels=self.feature_count,
                    hop_length=self.window_stride_samples,
                    n_fft=self.window_size_samples,
                    fmin=20,
                    fmax=8000)
        spectrogram = librosa.power_to_db(spectrogram)
        spectrogram = spectrogram.astype(np.float32)
        return spectrogram.flatten()
