from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import argparse
from six.moves import range
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
np.random.seed(0)

import tensorflow as tf
tf.set_random_seed(0)
from tensorflow.python.ops import seq2seq

from tensorflow.examples.tutorials.mnist import input_data
import keras.backend as K
from keras.layers import Dense

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import take_glimpses, accuracy_score, translate, plot_glimpse


def run(args):
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    num_steps = args.num_time_steps
    num_classes = 10
    num_lstm_units = args.num_lstm_units
    num_lstm_layer = 1
    alpha = args.alpha
    location_sigma = args.location_sigma
    glimpse_size = (12, 12)

    image_rows, image_cols = [int(v) for v in args.image_size.split("x")]

    mnist = input_data.read_data_sets("data", one_hot=True)

    sess = tf.Session()
    K.set_session(sess)

    image = tf.placeholder(tf.float32, (None, image_rows, image_cols, 1))
    label = tf.placeholder(tf.int32, (None, num_classes))

    tf.image_summary("translated mnist", image, max_images=3)

    cell = tf.nn.rnn_cell.LSTMCell(num_lstm_units, forget_bias=1., use_peepholes=True, state_is_tuple=True)
    # cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_lstm_layer, state_is_tuple=True)
    state = initial_state = cell.zero_state(tf.shape(image)[0], dtype=tf.float32)

    location_net = Dense(2, activation="linear", name="location_net")
    h_g = Dense(128, activation="relu", name="h_g")
    h_l = Dense(128, activation="relu", name="h_l")
    linear_h_g = Dense(256, activation="linear", name="linear_h_g")
    linear_h_l = Dense(256, activation="linear", name="linear_h_l")

    locations = []
    loc_means = []
    with tf.variable_scope("RNN"):
        for time_step in range(num_steps):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()

            h_tm1 = state.h

            loc_mean = location_net(h_tm1)
            tf.histogram_summary("loc_mean(t=%d) without tanh" % time_step, loc_mean)
            # loc_mean = 1.7159 * tf.nn.tanh(2/3 * loc_mean)
            # tf.histogram_summary("loc_mean(t=%d)" % time_step, loc_mean)
            locations += [tf.stop_gradient(tf.random_normal((batch_size, 2), loc_mean, location_sigma))]
            loc_means += [loc_mean]

            sizes = [(glimpse_size[0] * (i + 1), glimpse_size[1] * (i + 1))
                     for i in range(3)]
            glimpses = take_glimpses(image, locations[-1], sizes)
            glimpse = tf.concat(3, glimpses)
            glimpse = tf.reshape(glimpse, (-1, np.prod(glimpse_size) * len(sizes)))

            _h_g = h_g(glimpse)
            _h_l = h_l(locations[-1])
            inputs = tf.nn.relu(linear_h_g(_h_g) + linear_h_l(_h_l))
            (cell_output, state) = cell(inputs, state)
            tf.image_summary("12x12 glimpse t=%d" % time_step, glimpses[-1], max_images=5)

    logits = Dense(num_classes, name="logits")(state.h)
    inference = tf.nn.softmax(logits)
    prediction = tf.arg_max(inference, 1)
    R = tf.cast(tf.equal(prediction, tf.arg_max(label, 1)), tf.float32)
    R = tf.stop_gradient(tf.expand_dims(R, 1))

    accuracy = tf.reduce_mean(R)
    tf.scalar_summary("accuracy", accuracy)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(label, tf.float32))
    loss = tf.reduce_mean(loss)
    tf.scalar_summary("xentropy", loss)

    b = K.variable(0., name="baseline")
    tf.scalar_summary("baseline", b)

    reinforce_loss = 0.
    for time_step, (l, l_mean) in enumerate(zip(locations, loc_means)):
        b_val = 0.
        if args.baseline:
            b_val = tf.stop_gradient(b)

        p = 1. / tf.sqrt(2 * np.pi * tf.square(location_sigma))
        p *= tf.exp(-tf.square(l - l_mean) / (2 * tf.square(location_sigma)))
        reinforce_loss -= alpha * (R - b_val) * tf.log(p + K.epsilon())

    baseline_loss = tf.squared_difference(tf.reduce_mean(R), b)
    tf.scalar_summary("loss:baseline", baseline_loss)

    reinforce_loss = tf.reduce_sum(tf.reduce_mean(reinforce_loss, reduction_indices=0))
    tf.scalar_summary("loss:reinforce", reinforce_loss)

    total_loss = loss + reinforce_loss + baseline_loss
    tf.scalar_summary("loss:total", total_loss)

    if str.lower(args.optimizer) == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    elif str.lower(args.optimizer) == "momentum":
        optimizer = tf.train.MomentumOptimizer(learning_rate=args.learning_rate, momentum=args.momentum)

    tvars = tf.trainable_variables()
    grads = tf.gradients(total_loss, tvars)
    for tvar, grad in zip(tvars, grads):
        tf.histogram_summary(tvar.name, grad)
    train_step = optimizer.apply_gradients(zip(grads, tvars))

    merged = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(args.logdir, sess.graph)

    # Training
    sess.run(tf.initialize_all_variables())
    initial_c, initial_h = sess.run([initial_state.c, initial_state.h],
                                    feed_dict={image: np.zeros((batch_size, image_rows, image_cols, 1))})

    saver = tf.train.Saver()
    if args.train == 1:
        epoch_loss = []
        epoch_reinforce_loss = []
        epoch_acc = []

        global_step = 0
        while mnist.train.epochs_completed < num_epochs:
            current_epoch = mnist.train.epochs_completed
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = translate(batch_x.reshape((-1, 28, 28, 1)), size=(image_rows, image_cols))

            preds, loss, r_loss, summary, _ = sess.run([prediction, total_loss, reinforce_loss, merged, train_step],
                                                       feed_dict={image: batch_x, label: batch_y,
                                                                  initial_state.c: initial_c, initial_state.h: initial_h,
                                                                  K.learning_phase(): 1})
            epoch_loss += [loss]
            epoch_reinforce_loss += [r_loss]
            epoch_acc += [accuracy_score(preds, np.argmax(batch_y, axis=1))]

            summary_writer.add_summary(summary, global_step)
            global_step += 1

            if mnist.train.epochs_completed != current_epoch:
                print("[Epoch %d/%d]" % (current_epoch + 1, num_epochs))
                print("loss:", np.asarray(epoch_loss).mean())
                print("reinforce_loss: %.5f+/-%.5f",
                      np.asarray(reinforce_loss).mean(), np.asarray(reinforce_loss).std())
                print("acc: ", np.asarray(epoch_acc).mean())

                epoch_acc = []
                epoch_loss = []
                epoch_reinforce_loss = []

                val_loss = []
                val_reinforce_loss = []
                val_acc = []
                while mnist.validation.epochs_completed != 1:
                    batch_x, batch_y = mnist.validation.next_batch(batch_size)
                    batch_x = translate(batch_x.reshape((-1, 28, 28, 1)), size=(image_rows, image_cols))
                    res = sess.run([prediction, total_loss, reinforce_loss] + locations,
                                   feed_dict={image: batch_x.reshape((-1, image_rows, image_cols, 1)),
                                              label: batch_y,
                                              initial_state.c: initial_c, initial_state.h: initial_h,
                                              K.learning_phase(): 0})
                    preds, loss, r_loss = res[:3]
                    locs = res[3:]
                    val_loss += [loss]
                    val_reinforce_loss += [r_loss]
                    val_acc += [accuracy_score(preds, np.argmax(batch_y, axis=1))]

                    images = batch_x.reshape((-1, image_rows, image_cols))
                    locs = np.asarray(locs, dtype=np.float32)
                    locs = (locs + 1) * (image_rows / 2)
                    plot_glimpse(images, locs, name=args.logdir + "/glimpse.png")
                mnist.validation._epochs_completed = 0
                mnist.validation._index_in_epoch = 0

                print("Val loss:", np.asarray(val_loss).mean())
                print("Val reinforce_loss: %.5f+/-%.5f",
                      np.asarray(val_reinforce_loss).mean(), np.asarray(val_reinforce_loss).std())
                print("Val acc: ", np.asarray(val_acc).mean())
        saver.save(sess, "model.ckpt")

    if len(args.checkpoint) > 0:
        saver.restore(sess, args.checkpoint)

    # plot results
    batch_x, _ = mnist.train.next_batch(batch_size)
    batch_x = translate(batch_x.reshape((-1, 28, 28, 1)), size=(image_rows, image_cols))

    locs = sess.run(locations, feed_dict={image: batch_x.reshape((-1, image_rows, image_cols, 1)),
                                          initial_state.c: initial_c, initial_state.h: initial_h,
                                          K.learning_phase(): 0})

    images = batch_x.reshape((-1, image_rows, image_cols))
    locs = np.asarray(locs, dtype=np.float32)
    locs = (locs + 1) * (image_rows / 2)
    plot_glimpse(images, locs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", type=int, default=1)
    parser.add_argument("--checkpoint", dest="checkpoint", type=str, default="model.ckpt")
    parser.add_argument("--logdir", dest="logdir", type=str, default="logs")
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=64)
    parser.add_argument("--num_time_steps", dest="num_time_steps", type=int, default=5)
    parser.add_argument("--num_lstm_units", dest="num_lstm_units", type=int, default=256)
    parser.add_argument("--image_size", dest="image_size", type=str, default="128x128")
    parser.add_argument("--alpha", dest="alpha", type=float, default=1.0)
    parser.add_argument("--location_sigma", dest="location_sigma", type=float, default=0.01)
    parser.add_argument("--baseline", dest="baseline", action="store_true")
    parser.add_argument("--no-baseline", dest="baseline", action="store_false")
    parser.add_argument("--lr", dest="learning_rate", default=1e-3, type=float)
    parser.add_argument("--optimizer", dest="optimizer", default="momentum")
    parser.add_argument("--momentum", dest="momentum", default=0.9, type=float)
    args = parser.parse_args()

    assert str.lower(args.optimizer) in ["adam", "momentum"]

    if os.path.exists(args.logdir):
        shutil.rmtree(args.logdir)

    run(args)
