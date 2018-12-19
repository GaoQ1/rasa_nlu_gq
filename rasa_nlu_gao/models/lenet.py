from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
import time

# Build a convolutional neural network
def conv_net(x, n_classes, num_layers,layer_size, C2, dropout, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet'):
        # Flatten the data to a 1-D vector for the fully connected layer
        x = tf.contrib.layers.flatten(x)

        reg = tf.contrib.layers.l2_regularizer(C2)
        name = 'dense'
        for i in range(num_layers):
            x = tf.layers.dense(inputs=x,
                                units=layer_size[i],
                                activation=tf.nn.relu,
                                kernel_regularizer=reg,
                                name='hidden_layer_{}_{}'.format(name, i))
            x = tf.layers.dropout(x, rate=dropout, training=is_training)

        out = tf.layers.dense(inputs=x,
                            units=n_classes,
                            kernel_regularizer=reg,
                            name='dense_layer_{}'.format(name))
    return out