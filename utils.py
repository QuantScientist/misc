from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from keras.layers import Dense
from keras.preprocessing.image import apply_transform

import matplotlib.pyplot as plt


def take_glimpses(image, location, sizes):
    glimpses = []

    resize = sizes[0]
    for size in sizes:
        glimpse = tf.image.extract_glimpse(image, size=size, offsets=location,
                                           normalized=True, centered=True)
        glimpses += [tf.image.resize_images(glimpse, resize[0], resize[1])]

    return glimpses


def glimpse_network(image, location, sizes, activation="relu",
                    glimpse_num_features=128, location_num_features=128, output_dim=256):
    assert len(sizes) == 3

    with tf.variable_scope("glimpse_network"):
        glimpses = []

        resize = sizes[0]
        for size in sizes:
            glimpse = tf.image.extract_glimpse(image, size=size, offsets=location, uniform_noise=False,
                                               normalized=True, centered=True)
            glimpses += [tf.image.resize_images(glimpse, resize[0], resize[1])]

        glimpse = tf.concat(-1, glimpses)
        glimpse = tf.reshape(glimpse, (-1, np.prod(resize) * len(sizes)))
        glimpse_feature = Dense(glimpse_num_features, activation=activation)(glimpse)
        location_feature = Dense(location_num_features, activation=activation)(location)

        feature = Dense(output_dim, activation=activation)(glimpse_feature + location_feature)
        return feature, glimpses


def accuracy_score(y_preds, y_true):
    return np.sum((y_preds == y_true).astype(np.float32)) / y_preds.shape[0]


def translate(batch_x, size=(128, 128)):
    """Make translated mnist"""
    height = batch_x.shape[1]
    width = batch_x.shape[2]
    X = np.zeros((batch_x.shape[0],) + size + (1,), dtype=batch_x.dtype)
    X[:, :height, :width, :] = batch_x
    for i, x in enumerate(X[:]):
        tx = np.random.uniform(-(size[1] - width), 0)
        ty = np.random.uniform(-(size[0] - height), 0)

        translation_matrix = np.asarray([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ], dtype=batch_x.dtype)

        X[i, :, :, :] = apply_transform(x, translation_matrix, channel_index=2, fill_mode="nearest", cval=0.)
    return X


def plot_glimpse(images, locations, name="glimpse.png"):
    image = images[0]
    location = locations[:, 0, :]

    fig = plt.figure()
    plt.imshow(image, cmap=plt.get_cmap("gray"))
    plt.plot(location[:, 0], location[:, 1])
    for i, (x, y) in enumerate(location):
        plt.annotate("t=%d" % i, xy=(x, y), xytext=(-10, 10),
                     textcoords="offset points", ha="right", va="bottom",
                     bbox=dict(boxstyle="round, pad=0.5", fc="white", alpha=0.5),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=0"))
    plt.savefig(name)
    plt.gcf().clear()
    plt.close("all")
