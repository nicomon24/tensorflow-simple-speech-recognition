'''
    Load a model checkpoint to predict. Save to CSV
'''

from data_loader import TestDataLoader
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse, sys
from preprocessing import Preprocessor
from models import create_model
from tqdm import trange

def main(_):
    # Setting up tensorflow
    tf.logging.set_verbosity(tf.logging.INFO)
    sess = tf.InteractiveSession()
    # Input dataset
    preprocessor = Preprocessor(feature_count=40, window_size_ms=20, window_stride_ms=10)
    dl = TestDataLoader(FLAGS.data_dir, preprocessor)
    label_count = len(dl.classes)

    # Create input and call models to create the output tensors
    fingerprint_input = tf.placeholder(tf.float32, [None, preprocessor.fingerprint_size], name='fingerprint_input')
    fingerprint_input_4d = tf.reshape(fingerprint_input, [-1, preprocessor.feature_count, preprocessor.window_number, 1])
    logits = create_model(FLAGS.architecture, fingerprint_input_4d, {
        'label_count' : label_count
    }, is_training=False)
    predicted_indices = tf.argmax(logits, 1)

    # Init TF
    tf.global_variables_initializer().run()
    # Load checkpoint

    if FLAGS.start_checkpoint:
        _saver = tf.train.Saver(tf.global_variables())
        _saver.restore(sess, FLAGS.start_checkpoint)
    else:
        raise Exception('Need to provide a starting checkpoint for predictions')

    fnames = []
    predictions = []
    # Iterate through data
    for i in trange(0, len(dl.test_set), FLAGS.batch_size):
        _samples, _fnames = dl.get_test_data(FLAGS.batch_size, i)
        _predictions = sess.run(
            [predicted_indices],
            feed_dict={
                fingerprint_input: _samples,
            })[0]
        fnames.extend(_fnames)
        predictions.extend([dl.classes[k] for k in _predictions])
    # Write to CSV file the results
    df = pd.DataFrame({'fname' : fnames, 'label' : predictions})
    df.to_csv(FLAGS.output, index=False, header=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/speech_dataset/',
      help="""\
      Directory of files used for prediction.
      """)
    parser.add_argument('--architecture', type=str, default='vgg',
      help="""\
      Model architecture to use.
      """)
    parser.add_argument('--start_checkpoint', type=str, default='',
      help="""\
      Optimizer to use.
      """)
    parser.add_argument('--batch_size', type=int, default=128,
       help="""\
       Batch size.
       """)
    parser.add_argument('--output', type=str, default='output.csv',
       help="""\
       Where to store the results.
       """)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
