from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

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
                           initializer=tf.constant_initializer(1.0, dtype=tf.float32))


def build_discriminator(X, num_features, num_hidden):
    W = get_weight("W1", shape=(num_features, num_hidden[0]))
    b = get_bias("b1", shape=(num_hidden[0],))
    fc1 = tf.nn.relu(tf.add(tf.matmul(X, W), b))

    W = get_weight("W2", shape=(num_hidden[0], num_hidden[1]))
    b = get_bias("b2", shape=(num_hidden[1]))
    fc2 = tf.nn.relu(tf.add(tf.matmul(fc1, W), b))

    W = get_weight("W3", shape=(num_hidden[1], 1))
    b = get_bias("b3", shape=(1,))
    return tf.nn.sigmoid(tf.add(tf.matmul(fc2, W), b))


def build_generator(z, num_features, latent_dim, num_hidden):
    W = get_weight("W1", shape=(latent_dim, num_hidden[0]))
    b = get_bias("b1", shape=(num_hidden[0],))
    fc1 = tf.nn.relu(tf.add(tf.matmul(z, W), b))

    W = get_weight("W2", shape=(num_hidden[0], num_hidden[1]))
    b = get_bias("b2", shape=(num_hidden[1],))
    fc2 = tf.nn.relu(tf.add(tf.matmul(fc1, W), b))

    W = get_weight("W3", shape=(num_hidden[1], num_features))
    b = get_bias("b3", shape=(num_features,))

    return tf.nn.sigmoid(tf.add(tf.matmul(fc2, W), b))


def build_model(num_features, latent_dim, num_hidden):
    model = {"loss": None, "optimizer": None, "X": None, "y": None,
             "summary": {"generator": [], "discriminator": []}}
    model["X"] = X = tf.placeholder(tf.float32, shape=(None, num_features))
    model["z"] = z = tf.placeholder(tf.float32, shape=(None, latent_dim))
    model["latent_dim"] = latent_dim

    with tf.variable_scope("Generator"):
        model["G_z"] = G_z = build_generator(z, num_features, latent_dim, num_hidden["generator"])

    with tf.variable_scope("Discriminator"):
        D1 = build_discriminator(X, num_features, num_hidden["discriminator"])

    with tf.variable_scope("Discriminator", reuse=True):
        # Use D2 while generator is being trained
        D2 = build_discriminator(G_z, num_features, num_hidden["discriminator"])

    with tf.variable_scope("Discriminator", reuse=True):
        # Don't backpropagate through generator
        D3 = build_discriminator(tf.stop_gradient(G_z), num_features, num_hidden["discriminator"])

    model["log(D1)"] = tf.log(D1)
    model["log(1-D2)"] = tf.log(1-D2)
    model["log(1-D3)"] = tf.log(1-D3)
    model["summary"]["discriminator"] += [
        tf.scalar_summary("D/log(D(x))", tf.reduce_mean(model["log(D1)"])),
        tf.scalar_summary("D/log(1-D(z))", tf.reduce_mean(model["log(1-D3)"]))]
    model["summary"]["generator"] += [
        tf.scalar_summary("G/log(1-D(z))", tf.reduce_mean(model["log(1-D2)"]))]

    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)

    model["D_loss"] = -tf.reduce_mean(model["log(D1)"] + model["log(1-D3)"])
    model["D_grads"] = optimizer.compute_gradients(model["D_loss"])
    model["D_optimizer"] = optimizer.apply_gradients(model["D_grads"])
    model["summary"]["discriminator"] += [tf.scalar_summary("D/loss", model["D_loss"])]

    model["G_loss"] = tf.reduce_mean(model["log(1-D2)"])
    model["G_grads"] = optimizer.compute_gradients(model["G_loss"])
    # Disable training for discriminator
    grads = model["G_grads"]
    for i, grad in enumerate(grads[:]):
        g = grad[0]
        if grad[1].name.startswith("Discriminator"):
            g = None
        grads[i] = (g, grad[1])
    model["G_grads"] = grads
    model["G_optimizer"] = optimizer.apply_gradients(model["G_grads"])
    model["summary"]["generator"] += [tf.scalar_summary("G/loss", model["G_loss"])]

    return model


def make_batches(batch_size, num_examples):
    idx = list(range(num_examples))
    np.random.shuffle(idx)
    num_batches = int(math.ceil(num_examples / batch_size))
    batches = [idx[i * batch_size: (i + 1) * batch_size] for i in xrange(num_batches)]
    return batches


def sample_noise(low, high, shape):
    z = np.random.uniform(low, high, size=shape)
    return z


def train(model, X, config):
    sess = tf.Session()

    D_merged = tf.merge_summary(model["summary"]["discriminator"])
    G_merged = tf.merge_summary(model["summary"]["generator"])
    writer = tf.train.SummaryWriter(config["logdir"], sess.graph)

    sess.run(tf.initialize_all_variables())

    for epoch in xrange(config["num_epochs"]):
        epoch_cost_D = 0.
        epoch_cost_G = 0.

        batches = make_batches(config["batch_size"], X.shape[0])
        for i, batch in enumerate(batches):
            # Train discriminator. (Generator is fixed.)
            z = sample_noise(-1.0, 1.0, shape=(len(batch), model["latent_dim"]))
            summary_str, log_D_x, log_D_G_z, loss, _ = sess.run(
                [D_merged, model["log(D1)"], model["log(1-D3)"], model["D_loss"], model["D_optimizer"]],
                feed_dict={model["X"]: X[batch, :], model["z"]: z})
            assert not np.isnan(loss)
            epoch_cost_D += loss / config["num_epochs"]

            writer.add_summary(summary_str, i)

            # Train generator. (Discriminator is fixed.)
            z = sample_noise(-1.0, 1.0, shape=(len(batch), model["latent_dim"]))
            summary_str, log_D_G_z, loss, _ = sess.run(
                [G_merged, model["log(1-D2)"], model["G_loss"], model["G_optimizer"]],
                feed_dict={model["z"]: z})
            assert not np.isnan(loss)
            epoch_cost_G += loss / config["num_epochs"]

            writer.add_summary(summary_str, i)

        print("[Epoch %d] Discriminator cost = %.5f" % (epoch, epoch_cost_D))
        print("           Generator cost = %.5f" % epoch_cost_G)

    z = sample_noise(-1.0, 1.0, shape=(10, model["latent_dim"]))
    G_z = sess.run([model["G_z"]], feed_dict={model["z"]: z})[0]

    # DEBUG
    plt.imshow(G_z[0, :].reshape((28, 28)), cmap=plt.get_cmap("gray"))
    plt.show()


if __name__ == "__main__":
    mnist = input_data.read_data_sets("data/", one_hot=True)
    X = np.vstack([mnist.train.images,
                   mnist.validation.images,
                   mnist.test.images])

    config = dict()
    config["num_epochs"] = 20
    config["batch_size"] = 128
    config["logdir"] = "logs/"

    if os.path.exists(config["logdir"]):
        shutil.rmtree(config["logdir"])

    num_hidden = {
        "generator": [256, 256],
        "discriminator": [256, 256]
    }
    model = build_model(num_features=X.shape[1],
                        latent_dim=2,
                        num_hidden=num_hidden)
    train(model, X, config)
