from data_loader import TrainDataLoader
import tensorflow as tf
import numpy as np
import argparse, sys, os
from preprocessing import Preprocessor
from models import create_model
import pandas as pd

def main(_):
    # Setting up tensorflow
    tf.logging.set_verbosity(tf.logging.INFO)
    sess = tf.InteractiveSession()
    # Create preprocessor and data loader
    preprocessor = Preprocessor(feature_count=40, window_size_ms=20, window_stride_ms=10)
    dl = TrainDataLoader(FLAGS.data_dir, preprocessor)
    label_count = len(dl.classes)

    # Parse experiment settings
    settings = pd.read_csv(FLAGS.settings)

    # Create input and call models to create the output tensors
    fingerprint_input = tf.placeholder(tf.float32, [None, preprocessor.fingerprint_size], name='fingerprint_input')
    fingerprint_input_4d = tf.reshape(fingerprint_input, [-1, preprocessor.feature_count, preprocessor.window_number, 1])
    logits, dropout_prob = create_model(FLAGS.architecture, fingerprint_input_4d, {
        'label_count' : label_count
    }, is_training=True)

    # Create following tensors
    ground_truth_input = tf.placeholder(
      tf.float32, [None, label_count], name='groundtruth_input')

    # Cross entropy with summary
    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                    labels=ground_truth_input, logits=logits))
    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    # Declare optimizer and learning rate
    learning_rate_input = tf.placeholder(tf.float32, [], name='learning_rate_input')
    if FLAGS.optimizer == 'gd':
        train_step = tf.train.GradientDescentOptimizer(learning_rate_input).minimize(cross_entropy_mean)
    elif FLAGS.optimizer == 'adam':
        train_step = tf.train.AdamOptimizer(learning_rate_input).minimize(cross_entropy_mean)
    else:
        raise Exception('Optimizer not recognized')

    predicted_indices = tf.argmax(logits, 1)
    expected_indices = tf.argmax(ground_truth_input, 1)
    correct_prediction = tf.equal(predicted_indices, expected_indices)
    confusion_matrix = tf.confusion_matrix(expected_indices, predicted_indices, num_classes=label_count)
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)

    # Create global step
    global_step = tf.train.get_or_create_global_step()
    increment_global_step = tf.assign(global_step, global_step + 1)

    # Create saver to save model
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=40)

    # Merge summaries and write them to dir
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir +'/'+FLAGS.log_alias+'_train', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir +'/'+FLAGS.log_alias+'_validation')

    # Init global variables and start step
    tf.global_variables_initializer().run()
    start_step = 1

    # Check if there is a checkpoint to restore
    if FLAGS.start_checkpoint:
        _saver = tf.train.Saver(tf.global_variables())
        _saver.restore(sess, FLAGS.start_checkpoint)
        start_step = global_step.eval(session=sess)
    tf.logging.info('Training from step: %d ', start_step)

    # Saving graph
    tf.train.write_graph(sess.graph_def, FLAGS.train_dir, FLAGS.architecture + '.pbtxt')

    # Total steps
    total_steps = np.sum(list(settings.steps))

    for training_step in range(start_step, total_steps+1):
        # Get the current learning rate
        training_steps_sum = 0
        for i in range(len(list(settings.steps))):
          training_steps_sum += list(settings.steps)[i]
          if training_step <= training_steps_sum:
            # Get the settings
            current_settings = settings.iloc[i]
            break
        # Get the data
        samples, labels = dl.get_train_data(FLAGS.batch_size, unknown_percentage=0.1, silence_percentage=0.1,
                                noise_volume=current_settings.background_volume,
                                noise_frequency=current_settings.background_frequency_train,
                                time_shift_samples=current_settings.time_shift_samples,
                                time_shift_frequency=current_settings.time_shift_frequency_train)
        train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
        [
            merged_summaries, evaluation_step, cross_entropy_mean, train_step,
            increment_global_step
        ],
        feed_dict={
            fingerprint_input: samples,
            ground_truth_input: labels,
            learning_rate_input: current_settings.learning_rate,
            dropout_prob: 0.8
        })
        train_writer.add_summary(train_summary, training_step)
        tf.logging.info('Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
                        (training_step, current_settings.learning_rate, train_accuracy * 100,
                         cross_entropy_value))

        # Check if we need to check validation now
        is_last_step = (training_step == total_steps)
        if (training_step % FLAGS.evaluation_step == 0) or is_last_step:
            validation_size = dl.get_validation_size()
            total_accuracy = 0
            total_conf_matrix = None
            for i in range(0, validation_size, FLAGS.batch_size):
                validation_samples, validation_labels = dl.get_validation_data(FLAGS.batch_size, i,
                                unknown_percentage=0.1, silence_percentage=0.1,
                                noise_volume=current_settings.background_volume,
                                noise_frequency=current_settings.background_frequency_validation,
                                time_shift_samples=current_settings.time_shift_samples,
                                time_shift_frequency=current_settings.time_shift_frequency_validation)
                validation_summary, validation_accuracy, conf_matrix = sess.run(
                    [merged_summaries, evaluation_step, confusion_matrix],
                    feed_dict={
                        fingerprint_input: validation_samples,
                        ground_truth_input: validation_labels,
                        dropout_prob: 1.0
                    })
                validation_writer.add_summary(validation_summary, training_step)
                batch_size = min(FLAGS.batch_size, validation_size - i)
                total_accuracy += (validation_accuracy * batch_size) / validation_size
                if total_conf_matrix is None:
                    total_conf_matrix = conf_matrix
                else:
                    total_conf_matrix += conf_matrix
            tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
            tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                      (training_step, total_accuracy * 100, validation_size))
        # Save the model checkpoint periodically.
        if (training_step % FLAGS.save_step_interval == 0 or training_step == total_steps):
          checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.architecture + '.ckpt')
          tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
          saver.save(sess, checkpoint_path, global_step=training_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/speech_dataset/',
      help="""\
      Directory of files used for training.
      """)
    parser.add_argument('--architecture', type=str, default='vgg',
      help="""\
      Model architecture to use.
      """)
    parser.add_argument('--optimizer', type=str, default='gd',
      help="""\
      Optimizer to use.
      """)
    parser.add_argument('--summaries_dir', type=str, default='train_logs',
      help="""\
      Optimizer to use.
      """)
    parser.add_argument('--train_dir', type=str, default='/tmp/isaac',
      help="""\
      Directory to write checkpoints to.
      """)
    parser.add_argument('--log_alias', type=str, default='isaac',
      help="""\
      Optimizer to use.
      """)
    parser.add_argument('--start_checkpoint', type=str, default='',
      help="""\
      Optimizer to use.
      """)
    parser.add_argument('--settings', type=str, default='0.001',
       help="""\
       How many training steps.
       """)
    parser.add_argument('--batch_size', type=int, default=128,
       help="""\
       Batch size.
       """)
    parser.add_argument('--evaluation_step', type=int, default=400,
       help="""\
       How frequently we need to check the validation score.
       """)
    parser.add_argument('--save_step_interval', type=int, default=400,
       help="""\
       How frequently we need to save the checkpoint.
       """)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
