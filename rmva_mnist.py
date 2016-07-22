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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def glimpse_network(image, location):
    with tf.variable_scope("glimpse_network"):
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


def accuracy_score(y_preds, y_true):
    return np.sum((y_preds == y_true).astype(np.int32)) / y_preds.shape[0]


def run(args):
    batch_size = 128
    num_epochs = 1000
    num_steps = 5
    num_classes = 10
    num_lstm_units = 256
    num_lstm_layer = 1
    alpha = 1.0
    location_sigma = 0.01

    mnist = input_data.read_data_sets("data", one_hot=True)

    sess = tf.Session()

    image = tf.placeholder(tf.float32, (None, 28, 28, 1))
    label = tf.placeholder(tf.int32, (None, num_classes))

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_lstm_units, forget_bias=0., state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_lstm_layer, state_is_tuple=True)

    locations = []
    loc_means = []
    state = initial_state = cell.zero_state(tf.shape(image)[0], dtype=tf.float32)
    location_net = Dense(2, activation="tanh")
    with tf.variable_scope("RNN"):
        for time_step in range(num_steps):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()

            loc_mean = location_net(state[0].c)
            locations += [tf.random_normal((batch_size, 2), loc_mean, location_sigma)]
            loc_means += [loc_mean]

            inputs = glimpse_network(image, locations[-1])
            (cell_output, state) = cell(inputs, state)

    logits = Dense(num_classes)(state[0].c)
    inference = tf.nn.softmax(logits)
    prediction = tf.arg_max(inference, 1)
    reward = tf.cast(tf.equal(prediction, tf.arg_max(label, 1)), tf.float32)
    reward = tf.stop_gradient(tf.expand_dims(reward, 1))

    loss = tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(label, tf.float32))

    reinforce_loss = 0.
    for i, (loc, mean) in enumerate(zip(locations, loc_means)):
        p = 1. / tf.sqrt(2 * np.pi * np.square(location_sigma))
        p *= tf.exp(-tf.square(loc - mean) / (2 * np.square(location_sigma)))

        R = 0
        if (i + 1) == len(locations):  # for location_T-1
            R = reward
        reinforce_loss += tf.reduce_sum(alpha * R * tf.log(p), reduction_indices=[-1])

    total_loss = tf.reduce_mean(loss + reinforce_loss)

    optimizer = tf.train.AdamOptimizer()
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), 1.0)
    train_step = optimizer.apply_gradients(zip(grads, tvars))

    # Training
    sess.run(tf.initialize_all_variables())
    initial_c, initial_h = sess.run([initial_state[0].c, initial_state[0].h],
                                    feed_dict={image: np.zeros((batch_size, 28, 28, 1))})

    saver = tf.train.Saver()
    if args.train == 1:
        epoch_loss = []
        epoch_acc = []
        while mnist.train.epochs_completed < num_epochs:
            current_epoch = mnist.train.epochs_completed
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            preds, loss, _ = sess.run([prediction, total_loss, train_step],
                                      feed_dict={image: batch_x.reshape((-1, 28, 28, 1)), label: batch_y,
                                                 initial_state[0].c: initial_c, initial_state[0].h: initial_h})
            epoch_loss += [loss]
            epoch_acc += [accuracy_score(preds, np.argmax(batch_y, axis=1))]

            if mnist.train.epochs_completed != current_epoch:
                print("[Epoch %d/%d]" % (current_epoch + 1, num_epochs))
                print("loss:", np.asarray(epoch_loss).mean())
                print("acc: ", np.asarray(epoch_acc).mean())

                epoch_acc = []
                epoch_loss = []

                val_loss = []
                val_acc = []
                while mnist.validation.epochs_completed != 1:
                    batch_x, batch_y = mnist.validation.next_batch(batch_size)
                    preds, loss = sess.run([prediction, total_loss],
                                           feed_dict={image: batch_x.reshape((-1, 28, 28, 1)), label: batch_y,
                                                      initial_state[0].c: initial_c, initial_state[0].h: initial_h})
                    val_loss += [loss]
                    val_acc += [accuracy_score(preds, np.argmax(batch_y, axis=1))]
                mnist.validation._epochs_completed = 0
                mnist.validation._index_in_epoch = 0

                print("Val loss:", np.asarray(val_loss).mean())
                print("Val acc: ", np.asarray(val_acc).mean())


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
