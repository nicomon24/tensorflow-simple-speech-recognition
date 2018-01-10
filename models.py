'''
    Collection of models to use. Each one receives an input tensor as input,
    returns logits
'''

import tensorflow as tf

def create_model(name, input_tensor, settings, is_training):
    if name == 'vgg':
        return create_vgg_model(input_tensor, settings, is_training)

def dropout_if_training(input_tensor, dropout_prob, is_training):
    if is_training:
        input_tensor = tf.nn.dropout(input_tensor, dropout_prob)
    return input_tensor

def batchnorm_if_training(input_tensor, is_training):
    if is_training:
        input_tensor = tf.layers.batch_normalization(input_tensor)
    return input_tensor

def create_vgg_model(input_tensor, settings, is_training):
    x = input_tensor
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    else:
        dropout_prob = None
    # First layer, 8 3x3 filters, 2 conv + relu
    x = tf.layers.conv2d(x, filters=8, kernel_size=(3, 3), padding='SAME',
                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    #x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, filters=8, kernel_size=(3, 3), padding='SAME',
                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    #x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    # First pooling
    x = tf.layers.max_pooling2d(x, 2, 2, 'SAME')
    x = dropout_if_training(x, dropout_prob, is_training)
    # Second layer, 16 3x3
    x = tf.layers.conv2d(x, filters=16, kernel_size=(3, 3), padding='SAME',
                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    #x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, filters=16, kernel_size=(3, 3), padding='SAME',
                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    #x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    # Second pooling
    x = tf.layers.max_pooling2d(x, 2, 2, 'SAME')
    x = dropout_if_training(x, dropout_prob, is_training)
    # Third layer, 32 3x3
    x = tf.layers.conv2d(x, filters=32, kernel_size=(3, 3), padding='SAME',
                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    #x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, filters=32, kernel_size=(3, 3), padding='SAME',
                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    #x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    # Third pooling
    x = tf.layers.max_pooling2d(x, 2, 2, 'SAME')
    x = dropout_if_training(x, dropout_prob, is_training)
    # Fully connected
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, 512, kernel_initializer=tf.contrib.layers.xavier_initializer())
    x = tf.nn.relu(x)
    x = dropout_if_training(x, dropout_prob, is_training)
    x = tf.layers.dense(x, 256, kernel_initializer=tf.contrib.layers.xavier_initializer())
    x = tf.nn.relu(x)
    final_fc = tf.layers.dense(x, settings['label_count'])
    if is_training:
      return final_fc, dropout_prob
    else:
      return final_fc
