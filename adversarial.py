from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
# noinspection PyUnresolvedReferences
from six.moves import range

import numpy as np
np.random.seed(0)

import tensorflow as tf
tf.set_random_seed(0)
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
import math


def glorot_uniform(fanin, fanout):
    scale = np.sqrt(6 / (fanin + fanout))
    return tf.random_uniform_initializer(-scale, scale)


def get_weight(name, shape):
    return tf.get_variable(name, shape=shape,
                           initializer=glorot_uniform(shape[0], shape[1]))


def get_bias(name, shape):
    return tf.get_variable(name, shape=shape,
                           initializer=tf.constant_initializer(0.0, dtype=tf.float32))


def fc(name, inputs, input_dim, output_dim, activation="tanh"):
    with tf.variable_scope(name):
        activation_fn = None
        if activation == "sigmoid":
            activation_fn = tf.nn.sigmoid
        elif activation == "tanh":
            activation_fn = tf.nn.tanh
        elif activation == "relu":
            activation_fn = tf.nn.relu
        elif activation == "elu":
            activation_fn = tf.nn.elu
        elif activation == "linear":
            activation_fn = lambda x: x

        W = get_weight("W", shape=(input_dim, output_dim))
        b = get_bias("b", shape=(output_dim,))
        return activation_fn(tf.nn.bias_add(tf.matmul(inputs, W), b))


def build_discriminator(X, num_features, num_hidden):
    D = {}
    out = fc("fc1", X, input_dim=num_features, output_dim=num_hidden[0],
             activation="linear")
    D["fc1"] = out = tf.nn.tanh(batch_norm(out))
    D["md1"] = out = minibatch_discriminator("md1", out, num_hidden[0], num_hidden[0])
    D["dropout1"] = out = tf.nn.dropout(out, 0.2)

    for i in range(1, len(num_hidden)):
        out = fc("fc%d" % (i + 1), out,
                  input_dim=num_hidden[i-1], output_dim=num_hidden[i],
                  activation="linear")
        D["fc%d" % (i + 1)] = out = tf.nn.tanh(batch_norm(out))
        D["md%d" % (i + 1)] = out = minibatch_discriminator(
            "md%d" % (i+1), out, num_hidden[i], num_hidden[i])

    D["sigmoid"] = fc("sigmoid", out, input_dim=num_hidden[-1], output_dim=1,
                      activation="sigmoid")
    return D


def batch_norm(x, name=""):
    mean, var = tf.nn.moments(x, axes=[0])
    return tf.nn.batch_normalization(x, mean, var,
                                     offset=None, scale=None,
                                     variance_epsilon=0.0001, name=name)


def minibatch_discriminator(name, x, num_features, output_dim, C=5):
    # TODO: better name for C
    with tf.variable_scope(name):
        T = get_weight("T", shape=(num_features, output_dim, C))
        M = tf.matmul(x, tf.reshape(T, (num_features, output_dim * C)))
        M = tf.reshape(M, (-1, output_dim, C))
        diff = tf.abs(tf.expand_dims(M, 0) - tf.expand_dims(M, 1))
        l1norm = tf.reduce_sum(diff, -1)
        return tf.reduce_sum(tf.exp(-l1norm), 1) * 1e-3


def build_generator(z, num_features, latent_dim, num_hidden):
    G = {}
    out = fc("fc1", z, input_dim=latent_dim, output_dim=num_hidden[0],
             activation="linear")
    out = batch_norm(out)
    out = tf.nn.elu(out)
    G["fc1"] = out

    for i in range(1, len(num_hidden)):
        out = fc("fc%d" % (i + 1), out,
                 input_dim=num_hidden[i-1], output_dim=num_hidden[i],
                 activation="linear")
        out = batch_norm(out)
        out = tf.nn.elu(out)
        G["fc%d" % (i + 1)] = out

    out = fc("linear", out,
             input_dim=num_hidden[-1], output_dim=num_hidden[-1],
             activation="linear")
    out = batch_norm(out)
    out = fc("linear2", out,
             input_dim=num_hidden[-1], output_dim=num_features,
             activation="linear")
    # out = tf.nn.log(out)
    G["linear"] = out
    return G


def historical_averaging(vars, moving_averages):
    s = 0.
    for var, ma in zip(vars, moving_averages):
        s += tf.reduce_sum(tf.square(var - ma))
    return s


def build_model(num_features, latent_dim, num_hidden):
    model = {"loss": None, "optimizer": None, "X": None, "y": None, "ema": None,
             "summary": {"generator": [], "discriminator": []}}
    model["X"] = X = tf.placeholder(tf.float32, shape=(None, num_features))
    model["z"] = z = tf.placeholder(tf.float32, shape=(None, latent_dim))
    model["latent_dim"] = latent_dim
    model["ema"] = tf.train.ExponentialMovingAverage(decay=0.9999)

    with tf.variable_scope("Generator"):
        generator = build_generator(z, num_features, latent_dim, num_hidden["generator"])
        model["G_z"] = G_z = generator["linear"]
        model["summary"]["generator"] += [tf.histogram_summary("G/G_z", G_z)]

    with tf.variable_scope("Discriminator"):
        discriminator1 = build_discriminator(X, num_features, num_hidden["discriminator"])
        model["D(x)"] = D_X = discriminator1["sigmoid"]

    with tf.variable_scope("Discriminator", reuse=True):
        discriminator2 = build_discriminator(G_z, num_features, num_hidden["discriminator"])
        model["D(G(z))"] = D_G_z = discriminator2["sigmoid"]

    model["log(D(x))"] = tf.log(D_X)
    model["log(1-D(G(z)))"] = tf.log(1-D_G_z)
    model["log(D(G(z)))"] = tf.log(D_G_z)
    model["summary"]["discriminator"] += [
        tf.scalar_summary("D/log(D(x))", tf.reduce_mean(model["log(D(x))"])),
        tf.scalar_summary("D/log(1-D(G(z)))", tf.reduce_mean(model["log(1-D(G(z)))"]))]
    model["summary"]["generator"] += [
        tf.scalar_summary("G/log(1-D(G(z)))", tf.reduce_mean(model["log(1-D(G(z)))"]))]

    # Feature matching
    mean, var = tf.nn.moments(discriminator1["fc1"], axes=[0])
    mean_, var_ = tf.nn.moments(discriminator2["fc1"], axes=[0])
    model["feature_matching"] = tf.square(mean - mean_)

    optimizer = tf.train.AdamOptimizer()
    t_vars = tf.trainable_variables()
    D_vars = [v for v in t_vars if v.name.startswith("Discriminator")]
    G_vars = [v for v in t_vars if v.name.startswith("Generator")]

    model["D_maintain_ave_op"] = model["ema"].apply(D_vars)
    model["G_maintain_ave_op"] = model["ema"].apply(G_vars)
    D_moving_averages = [model["ema"].average(v) for v in D_vars]
    G_moving_averages = [model["ema"].average(v) for v in G_vars]
    model["D_historical_average"] = historical_averaging(D_vars, D_moving_averages)
    model["G_historical_average"] = historical_averaging(G_vars, G_moving_averages)

    model["D_loss"] = -tf.reduce_mean(model["log(D(x))"] + model["log(1-D(G(z)))"])
    model["D_loss"] += model["D_historical_average"]
    model["D_grads"] = optimizer.compute_gradients(model["D_loss"], var_list=D_vars)
    model["D_optimizer"] = optimizer.apply_gradients(model["D_grads"])
    model["summary"]["discriminator"] += [tf.scalar_summary("D/loss", model["D_loss"])]

    optimizer = tf.train.AdamOptimizer()
    # model["G_loss"] = tf.reduce_mean(model["log(1-D(G(z)))"] - model["log(D(G(z)))"])
    model["G_loss"] = tf.reduce_mean(model["feature_matching"])
    model["G_loss"] += model["G_historical_average"]
    model["G_grads"] = optimizer.compute_gradients(model["G_loss"], var_list=G_vars)
    model["G_optimizer"] = optimizer.apply_gradients(model["G_grads"])
    model["summary"]["generator"] += [tf.scalar_summary("G/loss", model["G_loss"])]

    # Add histograms of gradients to summary
    for grad, var in model["D_grads"]:
        model["summary"]["discriminator"] += [tf.histogram_summary("D/grads/" + var.name, grad)]

    for grad, var in model["G_grads"]:
        model["summary"]["discriminator"] += [tf.histogram_summary("G/grads/" + var.name, grad)]

    return model


def make_batches(batch_size, num_examples):
    idx = list(range(num_examples))
    np.random.shuffle(idx)
    num_batches = int(math.ceil(num_examples / batch_size))
    batches = [idx[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]
    return batches


def sample_noise(shape, low=-1.0, high=1.0):
    z = np.random.uniform(low, high, size=shape)
    return z


def train(sess, model, X, config):
    D_merged = tf.merge_summary(model["summary"]["discriminator"])
    G_merged = tf.merge_summary(model["summary"]["generator"])
    writer = tf.train.SummaryWriter(config["logdir"], sess.graph)

    sess.run(tf.initialize_all_variables())

    generated_samples = []

    z = sample_noise(shape=(config["batch_size"], model["latent_dim"]))
    G_z = sess.run(model["G_z"], feed_dict={model["z"]: z})
    generated_samples += [G_z]

    for epoch in range(config["num_epochs"]):
        epoch_cost_D = 0.
        epoch_cost_G = 0.

        batches = make_batches(config["batch_size"], X.shape[0])
        for i, batch in enumerate(batches):
            # Train discriminator. (Generator is fixed.)
            z = sample_noise(shape=(len(batch), model["latent_dim"]))
            summary_str, loss, _, _ = sess.run(
                [D_merged, model["D_loss"], model["D_optimizer"], model["D_maintain_ave_op"]],
                feed_dict={model["X"]: X[batch], model["z"]: z})
            assert not np.isnan(loss)
            epoch_cost_D += loss / len(batches)

            writer.add_summary(summary_str, i)

            if (i + 1) % 1 == 0:
                # Train generator. (Discriminator is fixed.)
                z = sample_noise(shape=(len(batch), model["latent_dim"]))
                summary_str, G_z, loss, _, _ = sess.run(
                    [G_merged, model["G_z"], model["G_loss"], model["G_optimizer"], model["G_maintain_ave_op"]],
                    feed_dict={model["X"]: X[batch], model["z"]: z})
                assert not np.isnan(loss)
                epoch_cost_G += loss / (len(batches) // 1)

                writer.add_summary(summary_str, i)

                # DEBUG
                generated_samples += [G_z]

        print("[Epoch %d] Discriminator cost = %.5f" % (epoch, epoch_cost_D))
        print("           Generator cost = %.5f" % epoch_cost_G)


    # DEBUG
    print("Plotting...")
    generated_samples = generated_samples[:10] + generated_samples[-10:]

    fig = plt.figure()
    sns.kdeplot(X.reshape(-1), label="p(x)")
    sns.kdeplot(generated_samples[-1].reshape(-1), label="p(G(z))")
    fig.savefig("result.png")

    fig = plt.figure()

    def init():
        sns.kdeplot(X.reshape(-1), label="p(x)")

    def update(i):
        sns.kdeplot(generated_samples[i].reshape(-1))

    ani = animation.FuncAnimation(fig, update,
                                  frames=len(generated_samples),
                                  init_func=init,
                                  repeat=False)
    ani.save("animation.gif", writer="imagemagick", fps=5)


if __name__ == "__main__":
    # mnist = input_data.read_data_sets("data/", one_hot=True)
    # X = np.vstack([mnist.train.images,
    #                mnist.validation.images,
    #                mnist.test.images])

    X = np.random.normal(size=10**5).reshape((-1, 1))

    config = dict()
    config["num_epochs"] = 20
    config["batch_size"] = 100
    config["logdir"] = "logs/"

    if os.path.exists(config["logdir"]):
        shutil.rmtree(config["logdir"])

    num_hidden = {
        "generator": [5] * 5,
        "discriminator": [32, 32]
    }

    sess = tf.Session()
    model = build_model(num_features=X.shape[1],
                        latent_dim=1,
                        num_hidden=num_hidden)
    train(sess, model, X, config)

    # DEBUG
    # z = sample_noise((1, 2))
    # G_z = sess.run(model["G_z"], feed_dict={model["z"]: z})
    # image = G_z.reshape((28, 28))
    # plt.imshow(image)
    # plt.show()
