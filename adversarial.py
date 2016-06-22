from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
from six.moves import range

import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
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
        elif activation == "linear":
            activation_fn = lambda x: x

        W = get_weight("W", shape=(input_dim, output_dim))
        b = get_bias("b", shape=(output_dim,))
        return activation_fn(tf.nn.bias_add(tf.matmul(inputs, W), b))


def build_discriminator(X, num_features, num_hidden):
    out = fc("fc1", X, input_dim=num_features, output_dim=num_hidden[0],
             activation="tanh")

    for i in range(len(num_hidden), 1):
        out = fc("fc%d" % (i + 1), out,
                 input_dim=num_hidden[i-1], output_dim=num_hidden[i],
                 activation="tanh")

    return fc("sigmoid", out, input_dim=num_hidden[-1], output_dim=1,
              activation="sigmoid")


def build_generator(z, num_features, latent_dim, num_hidden):
    out = fc("fc1", z, input_dim=latent_dim, output_dim=num_hidden[0],
             activation="tanh")

    for i in range(len(num_hidden), 1):
        out = fc("fc%d" % (i + 1), out,
                 input_dim=num_hidden[i-1], output_dim=num_hidden[i],
                 activation="tanh")

    return fc("linear", out, input_dim=num_hidden[-1], output_dim=num_features,
              activation="linear")


def build_model(num_features, latent_dim, num_hidden):
    model = {"loss": None, "optimizer": None, "X": None, "y": None,
             "summary": {"generator": [], "discriminator": []}}
    model["X"] = X = tf.placeholder(tf.float32, shape=(None, num_features))
    model["z"] = z = tf.placeholder(tf.float32, shape=(None, latent_dim))
    model["latent_dim"] = latent_dim

    with tf.variable_scope("Generator"):
        model["G_z"] = G_z = build_generator(z, num_features, latent_dim, num_hidden["generator"])

    with tf.variable_scope("Discriminator"):
        D_X = build_discriminator(X, num_features, num_hidden["discriminator"])

    with tf.variable_scope("Discriminator", reuse=True):
        D_G_z = build_discriminator(G_z, num_features, num_hidden["discriminator"])

    model["log(D(x))"] = tf.log(D_X)
    model["log(1-D(G(z)))"] = tf.log(1-D_G_z)
    model["log(D(G(z)))"] = tf.log(D_G_z)
    model["summary"]["discriminator"] += [
        tf.scalar_summary("D/log(D(x))", tf.reduce_mean(model["log(D(x))"])),
        tf.scalar_summary("D/log(1-D(G(z)))", tf.reduce_mean(model["log(1-D(G(z)))"]))]
    model["summary"]["generator"] += [
        tf.scalar_summary("G/log(1-D(G(z)))", tf.reduce_mean(model["log(1-D(G(z)))"]))]

    optimizer = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.9)
    t_vars = tf.trainable_variables()
    D_vars = [v for v in t_vars if v.name.startswith("Discriminator")]
    G_vars = [v for v in t_vars if v.name.startswith("Generator")]

    model["D_loss"] = -tf.reduce_mean(model["log(D(x))"] + model["log(1-D(G(z)))"])
    model["D_grads"] = optimizer.compute_gradients(model["D_loss"], var_list=D_vars)
    model["D_optimizer"] = optimizer.apply_gradients(model["D_grads"])
    model["summary"]["discriminator"] += [tf.scalar_summary("D/loss", model["D_loss"])]

    optimizer = tf.train.MomentumOptimizer(learning_rate=0.0003, momentum=0.9)
    model["G_loss"] = tf.reduce_mean(model["log(1-D(G(z)))"] - model["log(D(G(z)))"])
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

    for epoch in range(config["num_epochs"]):
        epoch_cost_D = 0.
        epoch_cost_G = 0.

        batches = make_batches(config["batch_size"], X.shape[0])
        for i, batch in enumerate(batches):
            # Train discriminator. (Generator is fixed.)
            z = sample_noise(shape=(len(batch), model["latent_dim"]))
            summary_str, log_D_x, log_D_G_z, loss, _ = sess.run(
                [D_merged, model["log(D(x))"], model["log(1-D(G(z)))"], model["D_loss"], model["D_optimizer"]],
                feed_dict={model["X"]: X[batch, :], model["z"]: z})
            assert not np.isnan(loss)
            epoch_cost_D += loss / len(batches)

            writer.add_summary(summary_str, i)

            if (i + 1) % 5 == 0:
                # Train generator. (Discriminator is fixed.)
                z = sample_noise(shape=(len(batch), model["latent_dim"]))
                summary_str, log_D_G_z, loss, _ = sess.run(
                    [G_merged, model["log(1-D(G(z)))"], model["G_loss"], model["G_optimizer"]],
                    feed_dict={model["z"]: z})
                assert not np.isnan(loss)
                epoch_cost_G += loss / (len(batches) // 10)

                writer.add_summary(summary_str, i)

        print("[Epoch %d] Discriminator cost = %.5f" % (epoch, epoch_cost_D))
        print("           Generator cost = %.5f" % epoch_cost_G)


if __name__ == "__main__":
    # mnist = input_data.read_data_sets("data/", one_hot=True)
    # X = np.vstack([mnist.train.images,
    #                mnist.validation.images,
    #                mnist.test.images])

    X = np.random.normal(size=10**5).reshape((-1, 1))

    config = dict()
    config["num_epochs"] = 50
    config["batch_size"] = 128
    config["logdir"] = "logs/"

    if os.path.exists(config["logdir"]):
        shutil.rmtree(config["logdir"])

    num_hidden = {
        "generator": [256, 256],
        "discriminator": [256, 256]
    }

    sess = tf.Session()
    model = build_model(num_features=X.shape[1],
                        latent_dim=1,
                        num_hidden=num_hidden)
    train(sess, model, X, config)

    # DEBUG
    z = sample_noise((10**5, 1))
    G_z = sess.run(model["G_z"], feed_dict={model["z"]: z})
    import seaborn as sns
    sns.kdeplot(G_z.reshape(-1), label="G(z)")
    sns.kdeplot(X.reshape(-1), label="x")
    plt.legend()
    plt.show()
