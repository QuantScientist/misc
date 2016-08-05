from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range

import numpy as np
np.random.seed(0)
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Layer, InputSpec, Input, Dense, Flatten, Reshape
from keras import initializations
from keras import activations
from keras.utils import np_utils
import keras.backend as K

from keras.utils.visualize_util import plot


class ConvLSTM(Layer):

    def __init__(self, nb_filter, nb_row, nb_col,
                 input_length=None, return_sequences=False, unroll=False,
                 init="glorot_uniform", inner_init="orthogonal",
                 forget_bias_init="one", activation="tanh",
                 inner_activation="hard_sigmoid", **kwargs):
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.return_sequences = return_sequences
        self.unroll = unroll
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        self.input_spec = [InputSpec(ndim=5)]
        self.input_length = input_length
        self.states = [None, None]
        super(ConvLSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        nb_input_channel = input_shape[2]

        W_shape = (self.nb_filter, nb_input_channel, self.nb_row, self.nb_col)
        U_shape = (self.nb_filter, self.nb_filter, self.nb_row, self.nb_col)
        C_shape = (self.nb_filter, input_shape[3], input_shape[4])
        b_shape = (self.nb_filter,)

        self.W_i = self.init(W_shape, name="{}_W_i".format(self.name))
        self.U_i = self.inner_init(U_shape, name="{}_U_i".format(self.name))
        self.C_i = self.inner_init(C_shape, name="{}_C_i".format(self.name))
        self.b_i = K.zeros(b_shape, name="{}_b_i".format(self.name))

        self.W_f = self.init(W_shape, name="{}_W_f".format(self.name))
        self.U_f = self.inner_init(U_shape, name="{}_U_f".format(self.name))
        self.C_f = self.inner_init(C_shape, name="{}_C_f".format(self.name))
        self.b_f = self.forget_bias_init(b_shape, name="{}_b_f".format(self.name))

        self.W_c = self.init(W_shape, name="{}_W_c".format(self.name))
        self.U_c = self.inner_init(U_shape, name="{}_U_c".format(self.name))
        self.b_c = K.zeros(b_shape, name="{}_b_c".format(self.name))

        self.W_o = self.init(W_shape, name="{}_W_o".format(self.name))
        self.U_o = self.inner_init(U_shape, name="{}_U_o".format(self.name))
        self.C_o = self.inner_init(C_shape, name="{}_C_o".format(self.name))
        self.b_o = K.zeros(b_shape, name="{}_b_o".format(self.name))

        self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                  self.W_c, self.U_c, self.b_c,
                                  self.W_f, self.U_f, self.b_f,
                                  self.W_o, self.U_o, self.b_o,
                                  self.C_i, self.C_f, self.C_o]

    def get_initial_states(self, x):
        initial_state = K.zeros_like(x)  # (samples, num_steps, input_channel, h, w)
        initial_state = K.sum(initial_state, [1, 2])  # (samples, h, w)
        initial_state = K.expand_dims(initial_state, 1)
        initial_state = K.repeat_elements(initial_state, self.nb_filter, 1)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def get_constants(self, x):
        return []

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])
        if self.return_sequences:
            return outputs
        else:
            return last_output

    def preprocess_input(self, x):
        return x

    def step(self, x, states):
        h_tm1 = states[0]
        c_tm1 = states[1]

        x_i = K.conv2d(x, self.W_i, border_mode="same")
        x_f = K.conv2d(x, self.W_f, border_mode="same")
        x_c = K.conv2d(x, self.W_c, border_mode="same")
        x_o = K.conv2d(x, self.W_o, border_mode="same")

        h_i = K.conv2d(h_tm1, self.U_i, border_mode="same")
        h_f = K.conv2d(h_tm1, self.U_f, border_mode="same")
        h_c = K.conv2d(h_tm1, self.U_c, border_mode="same")
        h_o = K.conv2d(h_tm1, self.U_o, border_mode="same")

        c_i = self.C_i * c_tm1
        c_f = self.C_f * c_tm1
        c_o = self.C_o * c_tm1

        b_i = K.reshape(self.b_i, (1, -1, 1, 1))
        b_f = K.reshape(self.b_f, (1, -1, 1, 1))
        b_c = K.reshape(self.b_c, (1, -1, 1, 1))
        b_o = K.reshape(self.b_o, (1, -1, 1, 1))

        i = self.inner_activation(x_i + h_i + c_i + b_i)
        f = self.inner_activation(x_f + h_f + c_f + b_f)
        c = f * c_tm1 + i * self.activation(x_c + h_c + b_c)
        o = self.inner_activation(x_o + h_o + c_o + b_o)
        h = o * self.activation(c)

        return h, [h, c]

    def get_output_shape_for(self, input_shape):
        # assume dim_ordering == th
        rows = input_shape[3]
        cols = input_shape[4]

        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.nb_filter, rows, cols)
        else:
            return (input_shape[0], self.nb_filter, rows, cols)

    def get_config(self):
        config = {"nb_filter": self.nb_filter,
                  "nb_row": self.nb_row,
                  "nb_col": self.nb_col,
                  "init": self.init,
                  "inner_init": self.inner_init,
                  "forget_bias_init": self.forget_bias_init,
                  "activation": self.activation}
        base_config = super(ConvLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
    nb_epoch = 1
    batch_size = 16
    nb_class = 10

    nb_time_step = 2
    nb_channel = 1
    img_rows = 28
    img_cols = 28

    inputs = Input(shape=(nb_time_step, nb_channel, img_rows, img_cols))
    x = ConvLSTM(8, 3, 3, return_sequences=False)(inputs)
    x = Flatten()(x)
    x = Dense(10, activation="softmax")(x)
    model = Model(input=inputs, output=x)
    model.compile(optimizer="sgd", loss="categorical_crossentropy",
                  metrics=["accuracy"])

    plot(model, show_shapes=True)

    (X_train, y_train), (X_val, y_val) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_val = X_val.reshape(X_val.shape[0], 1, img_rows, img_cols)
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_train /= 255
    X_val /= 255

    X_train = np.repeat(X_train[:, None, :, :, :], nb_time_step, 1)
    X_val = np.repeat(X_val[:, None, :, :, :], nb_time_step, 1)

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_val.shape[0], 'validation samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_class)
    Y_val = np_utils.to_categorical(y_val, nb_class)

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_data=(X_val, Y_val))
