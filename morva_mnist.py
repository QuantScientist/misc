from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

# noinspection PyUnresolvedReferences
from six.moves import range

import numpy as np
np.random.seed(0)

import tensorflow as tf
tf.set_random_seed(0)

from tensorflow.examples.tutorials.mnist import input_data

from keras.layers import Flatten, Convolution2D, Dense


def glimpse_net(image, location, input_size=(128, 128)):
    with tf.variable_scope("glimpse_network"):
        glimpses = []

        for size in [(8, 8), (16, 16), (32, 32)]:
            glimpse = tf.image.extract_glimpse(image, size, offsets=location, uniform_noise=False)
            glimpses += [tf.image.resize_images(glimpse, 8, 8)]

        glimpses = tf.concat(3, glimpses)

        # output: (batch_size, h - 2 * 3, w - 2 * 3, 32)
        g_image = Convolution2D(32, 3, 3, activation="relu", border_mode="valid")(glimpses)
        g_image = Convolution2D(32, 3, 3, activation="relu", border_mode="valid")(g_image)
        g_image = Convolution2D(32, 3, 3, activation="relu", border_mode="valid")(g_image)
        g_image = Flatten()(g_image)

        g_location = Dense(32 * (input_size[0] - 2 * 3) * (input_size[1] - 2 * 3), activation="relu")(location)
        g = g_image * g_location
        return g


def context_net(image, output_dim, input_size=(128, 128)):
    with tf.variable_scope("context_network"):
        ctx = Convolution2D(32, 3, 3, activation="relu", border_mode="valid")(image)
        ctx = Convolution2D(32, 3, 3, activation="relu", border_mode="valid")(ctx)
        ctx = Convolution2D(32, 3, 3, activation="relu", border_mode="valid")(ctx)
        ctx = Flatten()(ctx)
        ctx = Dense(output_dim, activation="relu")(ctx)
        return ctx


def emission_net(state):
    with tf.variable_scope("emission_network"):
        location_mean = Dense(2, activation="tanh")(state)
        return location_mean


def run(args):
    num_epochs = 1
    batch_size = 16
    num_time_steps = 0
    image_rows, image_cols = (28, 28)
    glimpse_size = (8, 8)
    num_classes = 10
    num_lstm_units = 128
    location_sigma = 0.01

    mnist = input_data.read_data_sets("data", one_hot=True)

    sess = tf.Session()

    image = tf.placeholder(tf.float32, (batch_size, image_rows, image_cols, 1))
    label = tf.placeholder(tf.int32, (batch_size, num_classes))

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_lstm_units, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2, state_is_tuple=True)

    locations = []
    location_means = []
    state = initial_state = cell.zero_state(batch_size, tf.float32)
    with tf.variable_scope("RNN") as scope:
        for t in range(num_time_steps):
            if t > 0:
                scope.reuse_variables()

            if t == 0:
                image_coarse = tf.image.resize_images(image, glimpse_size[0], glimpse_size[1])
                ctx = context_net(image_coarse, num_lstm_units, input_size=glimpse_size)
                state[1].c = ctx
                location_means[t] = emission_net(state[1].c)
                locations[t] = tf.random_normal((batch_size, 2), location_means[t], location_sigma)
                continue

            g = glimpse_net(image, locations[t-1], input_size=(image_rows, image_cols))
            _, state = cell(g, state)

            location_means[t] = emission_net(state[1].c)
            locations[t] = tf.random_normal((batch_size, 2), location_means[t], location_sigma)
    logits = Dense(num_classes)(state[0].c)
    inference = tf.nn.softmax(logits)
    prediction = tf.arg_max(inference, 1)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(label, tf.float32))

    reward = tf.cast(tf.equal(prediction, tf.arg_max(label, 1)), tf.float32)
    reward = tf.stop_gradient(tf.expand_dims(reward, 1))  # reward need to be treated as a constant

    reinforce_loss = 0.
    for t in range(num_time_steps):
        p = 1 / tf.sqrt(2 * np.pi * np.square(location_sigma))
        p *= tf.exp(-tf.square(locations[t] - location_means[t]) / (2 * np.square(location_sigma)))

        R = 0.
        if (t + 1) == num_time_steps:
            R = reward

        reinforce_loss += -tf.reduce_sum(R * tf.log(p), reduction_indices=[-1])
    total_loss = tf.reduce_mean(loss + reinforce_loss)

    optimizer = tf.train.AdamOptimizer()
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), 5.0)
    train_step = optimizer.apply_gradients(zip(grads, tvars))

    sess.run(tf.initialize_all_variables())
    initial_state_value = sess.run([initial_state[0].c, initial_state[0].h, initial_state[1].c, initial_state[1].h],
                                   feed_dict={image: np.zeros((batch_size, image_rows, image_cols, 1))})
    # WIP
    batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
    cost, _ = sess.run([total_loss, train_step],
                       feed_dict={image: batch_x.reshape((-1, 28, 28, 1)),
                                  label: batch_y,
                                  initial_state[0].c: initial_state_value[0],
                                  initial_state[0].h: initial_state_value[1],
                                  initial_state[1].c: initial_state_value[2],
                                  initial_state[1].h: initial_state_value[3]})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", default=1)
    args = parser.parse_args()
    run(args)
