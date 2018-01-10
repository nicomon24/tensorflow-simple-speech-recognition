'''
    Loads data and divide it into train set and validation set
'''
import os, re, random
from tensorflow.python.platform import gfile
from scipy.io.wavfile import read as read_wav
import numpy as np

BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
SILENCE_LABEL = 'silence'
UNKNOWN_LABEL = 'unknown'
WANTED_WORDS = ['yes','no','up','down','left','right','on','off','stop','go']
CLASSES = [SILENCE_LABEL, UNKNOWN_LABEL] + WANTED_WORDS

class TrainDataLoader:

    def __init__(self, data_dir, preprocessor, validation_percentage=10.0):
        self.data_dir = data_dir
        self.preprocessor = preprocessor
        self.create_data_index(validation_percentage)
        self.create_noise_samples()
        self.classes = CLASSES
        self.data_storage = DataStorage()

    # Load all the files and create an index
    def create_data_index(self, validation_percentage):
        self.silence_index = []
        self.data_index = []
        self.unknown_index = []
        # Scan the whole data directory
        search_path = os.path.join(self.data_dir, '*', '*.wav')
        for path in gfile.Glob(search_path):
            # Extract the subdirectory (class label)
            word = re.search('.*/([^/]+)/.*.wav', path).group(1).lower()
            if word == BACKGROUND_NOISE_DIR_NAME:
                self.silence_index.append(path)
            elif word not in CLASSES:
                self.unknown_index.append(path)
            else:
                self.data_index.append({
                    'path': path,
                    'label': word
                })
        # Shuffle into train and validation
        self.train_set = {
            'known' : [],
            'unknown' : [],
            'silence' : []
        }
        self.validation_set = {
            'known' : [],
            'unknown' : [],
            'silence' : []
        }
        # Partitioning known classes
        for audio in self.data_index:
            user_id = (audio['path'].split('/')[-1]).split('_nohash_')[0]
            random.seed(user_id)
            if random.random() < (validation_percentage / 100):
                self.validation_set['known'].append(audio)
            else:
                self.train_set['known'].append(audio)
        # Partitioning unknown classes
        for audio in self.unknown_index:
            user_id = (audio.split('/')[-1]).split('_nohash_')[0]
            random.seed(user_id)
            if random.random() < (validation_percentage / 100):
                self.validation_set['unknown'].append(audio)
            else:
                self.train_set['unknown'].append(audio)
        print('--- TRAIN ---')
        train_size = 0
        for key, value in self.train_set.items():
            print(key, len(value))
            train_size += len(value)
        print("Total train set size:", train_size)
        print('--- VALIDATION ---')
        val_size = 0
        for key, value in self.validation_set.items():
            print(key, len(value))
            val_size += len(value)
        print("Total validation set size:", val_size)

    def get_validation_size(self, unknown_percentage=0.1, silence_percentage=0.1):
        val_known_size = len(self.validation_set['known'])
        unknown_size = int(unknown_percentage * val_known_size)
        silence_size = int(silence_percentage * val_known_size)
        return val_known_size + unknown_size + silence_size

    def create_noise_samples(self):
        self.noises = []
        # Extract samples of 1s (length 16000) from the background noise index
        for f in self.silence_index:
            # Load file
            sample_rate, signal = read_wav(f)
            # Extract the more sample we can of length 1s
            # Remove sample_rate from len(signal) to avoid a shorter noise
            for i in range(0, len(signal) - sample_rate, sample_rate):
                self.noises.append(signal[i:i+sample_rate])

    # Return a batch from train dataset audios (randomly sampled)
    def get_train_data(self, batch_size, unknown_percentage=0.1,
                            silence_percentage=0.1, noise_frequency=1.0,
                            noise_volume=0.3, time_shift_frequency=0.5,
                            time_shift_samples=1600):
        batch_data = np.zeros((batch_size, self.preprocessor.fingerprint_size))
        labels = np.zeros((batch_size, len(CLASSES)))
        # Extract batch_size samples
        for i in range(batch_size):
            # Choose if the sample is known, unknown or silence
            pivot = random.random()
            if pivot < unknown_percentage:
                # Choosing an unknown audio file
                selected_path = np.random.choice(self.train_set['unknown'])
                labels[i, CLASSES.index(UNKNOWN_LABEL)] = 1
                sample_rate, signal = self.data_storage.get_item(selected_path)
            elif pivot < unknown_percentage + silence_percentage:
                # Choose a noise file
                labels[i, CLASSES.index(SILENCE_LABEL)] = 1
                signal = np.zeros(self.preprocessor.sample_length)
            else:
                # Choosing a known file
                selected_file = np.random.choice(self.train_set['known'])
                labels[i, CLASSES.index(selected_file['label'])] = 1
                sample_rate, signal = self.data_storage.get_item(selected_file['path'])
            # Check length and pad if necessary
            signal = self.preprocessor.check_audio_length(signal)
            # Timeshift
            if random.random() < time_shift_frequency:
                shift = np.random.randint(-time_shift_samples, time_shift_samples)
                signal_shifted = self.preprocessor.time_shift(signal, shift)
            else:
                signal_shifted = signal
            # Add noise
            if random.random() < noise_frequency:
                signal_with_noise = self.preprocessor.add_noise(signal_shifted, self.noises, k=noise_volume)
            else:
                signal_with_noise = signal_shifted
            # Preprocess and transform
            batch_data[i,:] = self.preprocessor.get_log_mel_spectrograms(signal_with_noise)
        return batch_data, labels

    # Return a batch from the validation dataset audios (sampled in order)
    def get_validation_data(self, batch_size, offset, unknown_percentage=0.1,
                            silence_percentage=0.1, noise_frequency=1.0,
                            noise_volume=0.3, time_shift_frequency=0.5,
                            time_shift_samples=1600):
        val_known_size = len(self.validation_set['known'])
        unknown_size = int(unknown_percentage * val_known_size)
        silence_size = int(silence_percentage * val_known_size)
        total_size = val_known_size + unknown_size + silence_size
        # Get the batch length (care for the end)
        real_batch_size = min(batch_size, total_size-offset)
        batch_data = np.zeros((real_batch_size, self.preprocessor.fingerprint_size))
        labels = np.zeros((real_batch_size, len(CLASSES)))
        # Extract batch_size samples
        for i in range(offset, offset + real_batch_size):
            # First get the knowns
            if i < val_known_size:
                selected_file = self.validation_set['known'][i]
                labels[i-offset, CLASSES.index(selected_file['label'])] = 1
                sample_rate, signal = self.data_storage.get_item(selected_file['path'])
            elif i < val_known_size + unknown_size:
                # Unknown
                selected_path = self.validation_set['unknown'][i - val_known_size]
                labels[i-offset, CLASSES.index(UNKNOWN_LABEL)] = 1
                sample_rate, signal = self.data_storage.get_item(selected_path)
            else:
                # Silence
                labels[i-offset, CLASSES.index(SILENCE_LABEL)] = 1
                signal = np.zeros(self.preprocessor.sample_length)
            # Check length and pad if necessary
            signal = self.preprocessor.check_audio_length(signal)
            # Timeshift
            if random.random() < time_shift_frequency:
                shift = np.random.randint(-time_shift_samples, time_shift_samples)
                signal_shifted = self.preprocessor.time_shift(signal, shift)
            else:
                signal_shifted = signal
            # Add noise
            if random.random() < noise_frequency:
                signal_with_noise = self.preprocessor.add_noise(signal_shifted, self.noises, k=noise_volume)
            else:
                signal_with_noise = signal_shifted
            # Featurize
            batch_data[i-offset,:] = self.preprocessor.get_log_mel_spectrograms(signal_with_noise)
        return batch_data, labels

class TestDataLoader:

    def __init__(self, data_dir, preprocessor):
        self.data_dir = data_dir
        self.preprocessor = preprocessor
        self.create_data_index()
        self.classes = CLASSES

    def create_data_index(self):
        self.test_set = []
        # Scan the whole data directory
        search_path = os.path.join(self.data_dir, '*.wav')
        for path in gfile.Glob(search_path):
            # Extract the subdirectory (class label)
            filename = path.split('/')[-1]
            self.test_set.append({
                'path' : path,
                'filename' : filename
            })

    def get_test_data(self, batch_size, offset):
        total_size = len(self.test_set)
        real_batch_size = min(batch_size, total_size-offset)
        batch_data = np.zeros((real_batch_size, self.preprocessor.fingerprint_size))
        filenames = []
        for i in range(offset, offset + real_batch_size):
            sample = self.test_set[i]
            # Load wav
            sr, signal = read_wav(sample['path'])
            filenames.append(sample['filename'])
            # Check signal length
            signal = self.preprocessor.check_audio_length(signal)
            # Featurize
            batch_data[i-offset,:] = self.preprocessor.get_log_mel_spectrograms(signal)
        return batch_data, filenames

class DataStorage:

    def __init__(self):
        self.data_storage = {}

    def get_item(self, key):
        try:
            return self.data_storage[key]
        except:
            # Lazy loading
            sr, signal = read_wav(key)
            self.data_storage[key] = (sr, signal)
            return sr, signal
