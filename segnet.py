from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
np.random.seed(0)

import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import ELU
from keras.layers import Permute
from keras.layers import Reshape


def get_channel_axis():
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == "th":
        channel_axis = 1
    else:
        channel_axis = -1
    return channel_axis


def build_autoencoder(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    ch_axis = get_channel_axis()

    z = Convolution2D(32, 3, 3, border_mode="same")(inputs)
    z = BatchNormalization(axis=ch_axis)(z)
    z = Activation("relu")(z)
    z = MaxPooling2D((2, 2))(z)

    z = Convolution2D(64, 3, 3, border_mode="same")(z)
    z = BatchNormalization(axis=ch_axis)(z)
    z = Activation("relu")(z)
    encoded = MaxPooling2D((2, 2))(z)

    z = Convolution2D(64, 3, 3, border_mode="same")(encoded)
    z = BatchNormalization(axis=ch_axis)(z)
    z = Activation("relu")(z)
    z = UpSampling2D((2, 2))(z)

    z = Convolution2D(32, 3, 3, border_mode="same")(z)
    z = BatchNormalization(axis=ch_axis)(z)
    z = Activation("relu")(z)
    z = UpSampling2D((2, 2))(z)

    z = Convolution2D(3, 3, 3, border_mode="same")(z)
    z = BatchNormalization(axis=ch_axis)(z)
    z = Activation("relu")(z)

    z = Convolution2D(num_classes, 1, 1, border_mode="same")(z)

    num_features = input_shape[0] * input_shape[1]
    if K.image_dim_ordering() == "th":
        z = Permute([2, 3, 1])(z)
        num_features = input_shape[1] * input_shape[2]

    z = Reshape((num_features, num_classes))(z)
    decoded = Activation("softmax")(z)

    autoencoder = Model(input=inputs, output=decoded)
    return autoencoder


if __name__ == "__main__":
    num_classes = 21
    ae = build_autoencoder(input_shape=(3, 32, 32), num_classes=num_classes)
    ae.compile(optimizer="adam",
               loss="categorical_crossentropy",
               metrics=["accuracy"])
    ae.fit(np.zeros((128, 3, 32, 32)), np.zeros((128, 32 * 32, num_classes)),
           batch_size=32, nb_epoch=1)
