from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
os.environ["KERAS_BACKEND"] = "tensorflow"

# noinspection PyUnresolvedReferences
from six.moves import range

import numpy as np
np.random.seed(0)

import tensorflow as tf
tf.set_random_seed(0)

from tensorflow.examples.tutorials.mnist import input_data

import keras.backend as K
from keras.models import Model
from keras.layers import Flatten, Convolution2D, Dense, Input, Activation, Lambda

import matplotlib
matplotlib.use("Agg")

from utils import take_glimpses, translate, plot_glimpse


def build_context_net(output_dim, glimpse_size):
    I_coarse = Input(shape=(glimpse_size[0], glimpse_size[1], 1))
    z = Convolution2D(64, 5, 5, activation="relu", dim_ordering="tf")(I_coarse)
    z = Convolution2D(64, 3, 3, activation="relu", dim_ordering="tf")(z)
    z = Convolution2D(128, 3, 3, activation="relu", dim_ordering="tf")(z)
    z = Flatten()(z)
    z = Dense(output_dim, activation="relu")(z)
    context_net = Model(input=I_coarse, output=z, name="context_net")
    return context_net


def build_glimpse_net(output_dim, glimpse_size):
    x = Input(shape=(glimpse_size[0], glimpse_size[1], 3))
    l = Input(shape=(2,))

    g_x = Convolution2D(64, 5, 5, activation="relu", dim_ordering="tf")(x)
    g_x = Convolution2D(64, 3, 3, activation="relu", dim_ordering="tf")(g_x)
    g_x = Convolution2D(128, 3, 3, activation="relu", dim_ordering="tf")(g_x)
    g_x = Flatten()(g_x)
    g_x = Dense(output_dim, activation="relu")(g_x)

    g_l = Dense(output_dim, activation="relu")(l)

    g = Lambda(lambda tensors: tf.mul(tensors[0], tensors[1]),
               output_shape=(output_dim,))([g_x, g_l])
    glimpse_net = Model(input=[x, l], output=g, name="glimpse_net")
    return glimpse_net


def build_emission_net(num_lstm_units):
    h2 = Input(shape=(num_lstm_units,))
    l_next = Dense(2, activation="linear")(h2)
    emission_net = Model(input=h2, output=l_next, name="emission_net")
    return emission_net


def run(args):
    num_epochs = args.num_epochs
    num_time_steps = args.N * args.S
    batch_size = args.batch_size
    image_rows, image_cols = args.image_size
    glimpse_size = [int(s) for s in args.glimpse_size.split("x")]
    num_classes = 10
    num_lstm_units = args.num_lstm_units
    location_sigma = args.location_sigma

    mnist = input_data.read_data_sets("data", one_hot=True)

    sess = tf.Session()
    K.set_session(sess)

    image = tf.placeholder(tf.float32, (None, image_rows, image_cols, 1))
    label = tf.placeholder(tf.int32, (None, args.S, num_classes))

    tf.image_summary("translated", image, max_images=3)

    # === Recurrent network ===
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_lstm_units, forget_bias=1., use_peepholes=True, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2, state_is_tuple=True)
    state = initial_state = cell.zero_state(batch_size, tf.float32)

    context_net = build_context_net(num_lstm_units, glimpse_size)
    glimpse_net = build_glimpse_net(num_lstm_units, glimpse_size)
    emission_net = Dense(2, name="emission_net")
    classification_net = Dense(num_classes, name="classification_net")
    baseline_net = Dense(1, name="baseline_net")

    image_coarse = tf.image.resize_images(image, glimpse_size)
    state2 = tf.nn.rnn_cell.LSTMStateTuple(c=state[1].c, h=context_net(image_coarse))
    state = (state[0], state2)

    y_preds = [None] * num_time_steps

    location_means = [emission_net(state[1].h)]
    locations = [tf.random_normal((batch_size, 2), location_means[-1], location_sigma)]

    rewards = []
    baselines = []
    loss = 0.
    accuracy = 0.

    glimpse_sizes = [(glimpse_size[0] * (i + 1), glimpse_size[1] * (i + 1))
                     for i in range(3)]
    with tf.variable_scope("RNN") as scope:
        for t in range(num_time_steps):
            if t > 0:
                scope.reuse_variables()
            baselines.append(baseline_net(tf.stop_gradient(state[1].h)))

            glimpses = take_glimpses(image, locations[-1] / args.ratio, glimpse_sizes)
            tf.image_summary("glimpse(t=%d)" % t, glimpses[0], max_images=3)
            glimpse = tf.concat(3, glimpses)

            g = glimpse_net([glimpse, locations[-1]])
            output, state = cell(g, state)

            location_means.append(emission_net(state[1].h))
            locations.append(tf.random_normal((batch_size, 2), mean=location_means[-1], stddev=location_sigma))

            if (t + 1) % args.N == 0:
                target_idx = t // args.N
                logits = classification_net(state[0].h)
                label_t = tf.cast(label[:, target_idx, :], tf.float32)
                y_preds[t] = tf.argmax(tf.nn.softmax(logits), 1)

                cumulative = 0.
                if len(rewards) > 0:
                    cumulative = rewards[t-args.N]
                reward = tf.cast(tf.equal(y_preds[t], tf.argmax(label_t, 1)), tf.float32)
                rewards.append(cumulative + tf.expand_dims(reward, 1))
                loss += tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits, label_t))
                accuracy += tf.reduce_mean(reward) / args.S

    reinforce_loss = 0.
    baseline_loss = 0.
    for t in range(num_time_steps):  # t = 0..T-1
        p = 1 / tf.sqrt(2 * np.pi * tf.square(location_sigma))
        p *= tf.exp(-tf.square(locations[t] - location_means[t]) / (2 * tf.square(location_sigma)))
        R = tf.stop_gradient(rewards[t // args.N])
        b = baselines[t]
        b_ = tf.stop_gradient(b)
        log_p = tf.log(p)
        tf.histogram_summary("p(t=%d)" % t, p)
        tf.histogram_summary("log_p(t=%d)" % t, log_p)
        reinforce_loss -= args.alpha * (R - b_) * log_p
        baseline_loss += tf.reduce_mean(tf.squared_difference(tf.reduce_mean(R), b))

    reinforce_loss = tf.reduce_sum(tf.reduce_mean(reinforce_loss, reduction_indices=0))
    total_loss = loss + reinforce_loss + baseline_loss
    tf.scalar_summary("loss:total", total_loss)
    tf.scalar_summary("loss:xentropy", loss)
    tf.scalar_summary("loss:reinforcement", reinforce_loss)
    tf.scalar_summary("loss:baseline", baseline_loss)
    tf.scalar_summary("accuracy", accuracy)

    optimizer = args.optimizer
    tvars = tf.trainable_variables()
    # grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), 5.0)
    grads = tf.gradients(total_loss, tvars)
    for tvar, grad in zip(tvars, grads):
        tf.histogram_summary(tvar.name, grad)
    train_step = optimizer.apply_gradients(zip(grads, tvars))

    merged = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(args.logdir, sess.graph)

    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver()
    if not args.test:
        epoch_loss = []
        epoch_reinforce_loss = []
        epoch_acc = []

        global_step = 0
        while mnist.train.epochs_completed < num_epochs:
            current_epoch = mnist.train.epochs_completed
            batch_x, batch_y = mnist.train.next_batch(args.batch_size)
            batch_x = translate(batch_x.reshape((-1, 28, 28, 1)), size=(image_rows, image_cols))
            batch_y = batch_y.reshape((-1, args.S, num_classes))

            acc, loss, r_loss, summary, _ = sess.run([accuracy, total_loss, reinforce_loss, merged, train_step],
                                                     feed_dict={image: batch_x, label: batch_y,
                                                     K.learning_phase(): 1})

            epoch_loss.append(loss)
            epoch_reinforce_loss.append(r_loss)
            epoch_acc.append(acc)

            summary_writer.add_summary(summary, global_step)
            global_step += 1

            if mnist.train.epochs_completed != current_epoch:
                val_loss = []
                val_reinforce_loss = []
                val_acc = []

                while mnist.validation.epochs_completed != 1:
                    batch_x, batch_y = mnist.validation.next_batch(batch_size)
                    batch_x = translate(batch_x.reshape((-1, 28, 28, 1)), size=(image_rows, image_cols))
                    batch_y = batch_y.reshape((-1, args.S, num_classes))
                    res = sess.run([accuracy, total_loss, reinforce_loss] + locations,
                                   feed_dict={image: batch_x,
                                              label: batch_y,
                                              K.learning_phase(): 0})
                    acc, loss, r_loss = res[:3]
                    locs = res[3:]
                    val_loss.append(loss)
                    val_reinforce_loss.append(r_loss)
                    val_acc.append(acc)

                    images = batch_x.reshape((-1, image_rows, image_cols))
                    locs = np.asarray(locs, dtype=np.float32)
                    locs = (locs + 1) * (image_rows / 2)
                    plot_glimpse(images, locs / args.ratio, name=args.logdir + "/glimpse.png")

                print("[Epoch %d/%d]" % (current_epoch + 1, num_epochs))
                print("loss:", np.asarray(epoch_loss).mean())
                print("reinforce_loss: %.5f+/-%.5f" % (
                    np.asarray(epoch_reinforce_loss).mean(),
                    np.asarray(epoch_reinforce_loss).std()))
                print("accuracy:", np.asarray(epoch_acc).mean())

                print("val_loss:", np.asarray(val_loss).mean())
                print("val_reinforce_loss:", np.asarray(val_reinforce_loss).mean())
                print("val_acc:", np.asarray(val_acc).mean())
                mnist.validation._epochs_completed = 0
                mnist.validation._index_in_epoch = 0

        saver.save(sess, args.checkpoint)

    if len(args.checkpoint) > 0:
        saver.restore(sess, args.checkpoint)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", dest="test", action="store_true", default=False)
    parser.add_argument("--checkpoint", dest="checkpoint", default="model.ckpt")
    parser.add_argument("--logdir", dest="logdir", type=str, default="logs")
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=1)
    parser.add_argument("--image_size_per_digit", dest="image_size", default="60x60")
    parser.add_argument("--S", dest="S", default=1, help="num targets", type=int)
    parser.add_argument("--N", dest="N", default=2, help="num steps per target", type=int)
    parser.add_argument("--num_lstm_units", dest="num_lstm_units", type=int, default=256)
    parser.add_argument("--location_sigma", dest="location_sigma", type=float, default=0.01)
    parser.add_argument("--lr", dest="lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", dest="optimizer", default="adam")
    parser.add_argument("--momentum", dest="momentum", default=0.9, type=float)
    parser.add_argument("--alpha", dest="alpha", type=float, default=1.)
    parser.add_argument("--baseline", dest="baseline", action="store_true", default=False)
    parser.add_argument("--glimpse_size", dest="glimpse_size", default="8x8")
    parser.add_argument("--ratio", dest="ratio", type=float, default=1.)
    args = parser.parse_args()

    args.image_size = [int(v) for v in args.image_size.split("x")]

    if os.path.exists(args.logdir):
        shutil.rmtree(args.logdir)

    if str.lower(args.optimizer) == "adam":
        args.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
    elif str.lower(args.optimizer) == "momentum":
        args.optimizer = tf.train.MomentumOptimizer(learning_rate=args.lr,
                                                    momentum=args.momentum)
    else:
        args.optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.lr)

    assert args.S == 1
    run(args)
