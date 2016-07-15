from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from six.moves import range
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
np.random.seed(0)

import tensorflow as tf
tf.set_random_seed(0)

from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import Dense

import matplotlib.pyplot as plt



def glimpse_network(image, location):
    glimpses = []

    glimpse = tf.image.extract_glimpse(image, size=[2, 2], offsets=location, uniform_noise=False)
    glimpses += [tf.image.resize_images(glimpse, 8, 8)]

    glimpse = tf.image.extract_glimpse(image, size=[4, 4], offsets=location, uniform_noise=False)
    glimpses += [tf.image.resize_images(glimpse, 8, 8)]

    glimpse = tf.image.extract_glimpse(image, size=[8, 8], offsets=location, uniform_noise=False)
    glimpses += [tf.image.resize_images(glimpse, 8, 8)]

    glimpses = tf.concat(3, glimpses)
    glimpses = tf.reshape(glimpses, (-1, 8 * 8 * 3))

    glimpse_feature = Dense(128, activation="relu")(glimpses)
    location_feature = Dense(128, activation="relu")(location)

    feature = Dense(256, activation="relu")(glimpse_feature + location_feature)
    return feature


def run(args):
    batch_size = 128
    num_iterations = 50000 // 128 * 50
    num_steps = 5
    num_classes = 10
    num_lstm_units = 128
    num_lstm_layer = 1
    alpha = 0.1
    loc_var = 0.3

    mnist = input_data.read_data_sets("data", one_hot=True)

    sess = tf.Session()

    image = tf.placeholder(tf.float32, (None, 28, 28, 1))
    label = tf.placeholder(tf.int32, (None, num_classes))

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_lstm_units, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_lstm_layer, state_is_tuple=True)

    locations = [tf.random_uniform((batch_size, 2), minval=-1, maxval=1)]
    loc_params = []
    state = initial_state = cell.zero_state(tf.shape(image)[0], dtype=tf.float32)
    with tf.variable_scope("RNN"):
        for time_step in range(num_steps):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()

            inputs = glimpse_network(image, locations[-1])
            (cell_output, state) = cell(inputs, state)

            loc_mean = Dense(2, activation="tanh")(state[0].h)
            locations += [tf.random_normal((batch_size, 2), loc_mean, loc_var)]

            loc_params += [(loc_var, loc_mean)]

    logits = Dense(10)(state[0].h)
    inference = tf.nn.softmax(logits)
    prediction = tf.arg_max(inference, 1)
    reward = tf.cast(tf.equal(prediction, tf.arg_max(label, 1)), tf.float32)
    reward = tf.expand_dims(reward, 1)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(label, tf.float32))

    reinforce_loss = 0.
    for i, (mean, log_var) in enumerate(loc_params):
        p = 1. / tf.sqrt(2 * np.pi * log_var)
        p *= tf.exp(-tf.square(locations[i] - mean) / (2 * log_var))
        if (i + 1) < (len(loc_params) - 1):  # if t <= T - 2
            R = 0.
        elif (i + 1) == (len(loc_params) - 1):  # if t = T - 1
            R = reward
        else:
            break
        reinforce_loss += -tf.reduce_sum(alpha * R * tf.log(p), reduction_indices=[-1])

    total_loss = tf.reduce_mean(loss + reinforce_loss)

    optimizer = tf.train.AdamOptimizer()
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), 5.0)
    train_step = optimizer.apply_gradients(zip(grads, tvars))

    # Training
    sess.run(tf.initialize_all_variables())
    initial_c, initial_h = sess.run([initial_state[0].c, initial_state[0].h],
                                    feed_dict={image: np.zeros((batch_size, 28, 28, 1))})

    saver = tf.train.Saver()
    if args.train == 1:
        for i in range(num_iterations):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            assert batch_x.shape[0] == batch_size
            preds, loss, _ = sess.run([prediction, total_loss, train_step],
                                      feed_dict={image: batch_x.reshape((-1, 28, 28, 1)), label: batch_y,
                                                 initial_state[0].c: initial_c, initial_state[0].h: initial_h})

            if i % 100 == 0:
                print("loss:", loss)
                print("acc:", np.sum(preds.astype(np.int32) == np.argmax(batch_y, axis=1)) / batch_size * 100)
        saver.save(sess, "model.ckpt")

    if len(args.checkpoint) > 0:
        saver.restore(sess, args.checkpoint)

    # plot results
    batch_x, _ = mnist.train.next_batch(batch_size)
    batch_x = batch_x.reshape((-1, 28, 28, 1))

    locs = sess.run(locations, feed_dict={image: batch_x.reshape((-1, 28, 28, 1)),
                                          initial_state[0].c: initial_c, initial_state[0].h: initial_h})

    img = batch_x[0].reshape(28, 28)
    locs = np.asarray(locs, dtype=np.float32)[:, 0, :]
    locs = (locs + 1) * (28 / 2)

    fig = plt.figure()
    plt.imshow(img, cmap=plt.get_cmap("gray"))
    plt.plot(locs[:, 0], locs[:, 1])
    for i, (x, y) in enumerate(locs):
        plt.annotate("t=%d" % i, xy=(x, y), xytext=(-10, 10),
                     textcoords="offset points", ha="right", va="bottom",
                     bbox=dict(boxstyle="round, pad=0.5", fc="white", alpha=0.5),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=0"))
    plt.savefig("result.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", type=int, default=1)
    parser.add_argument("--checkpoint", dest="checkpoint", type=str, default="")
    args = parser.parse_args()
    run(args)
