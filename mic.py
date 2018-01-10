import pyaudio
import tensorflow as tf
import numpy as np
import argparse, sys, os
from preprocessing import Preprocessor
from data_loader import CLASSES
from models import create_model
from collections import deque

CHUNK = 1600
CHANNELS = 1
RATE = 16000

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='',
  help="""\
  Directory of files used for training.
  """)
parser.add_argument('--architecture', type=str, default='vgg',
  help="""\
  Model architecture to use.
  """)
FLAGS, unparsed = parser.parse_known_args()

print("Loading model...")

tf.logging.set_verbosity(tf.logging.INFO)
sess = tf.InteractiveSession()
preprocessor = Preprocessor(feature_count=40, window_size_ms=20, window_stride_ms=10)

fingerprint_input = tf.placeholder(tf.float32, [None, preprocessor.fingerprint_size], name='fingerprint_input')
fingerprint_input_4d = tf.reshape(fingerprint_input, [-1, preprocessor.feature_count, preprocessor.window_number, 1])
logits = create_model(FLAGS.architecture, fingerprint_input_4d, {
    'label_count' : len(CLASSES)
}, is_training=False)
predicted_indices = tf.argmax(logits, 1)

tf.global_variables_initializer().run()
if FLAGS.model:
    _saver = tf.train.Saver(tf.global_variables())
    _saver.restore(sess, FLAGS.model)
else:
    raise Exception('Need to provide a starting checkpoint for running')

print('Initializing microphone...')

p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paInt32,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK)
print('Recording...')

signal_cum = []

while True:
    # Extract a sample of CHUNK length
    data = stream.read(CHUNK)
    signal = np.fromstring(data, np.int32);
    # Append to current window (queue)
    signal_cum.extend(signal)
    signal_cum = signal_cum[-16000:]
    # If length of the queue is sufficient, classify
    signal_padded = preprocessor.check_audio_length(np.array(signal_cum))
    features = np.reshape(preprocessor.get_log_mel_spectrograms(signal_padded), (1, -1))
    prediction = sess.run(
        predicted_indices,
        feed_dict={
            fingerprint_input: features,
        })
    # Check results, print command if found
    predicted_class = CLASSES[prediction[0]]
    print("RECOGNIZED COMMAND: ", predicted_class)

stream.stop_stream()
stream.close()

p.terminate()
