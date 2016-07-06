from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# noinspection PyUnresolvedReferences
from six.moves import range

import numpy as np
np.random.seed(0)

import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import ELU
from keras.layers import LeakyReLU

import matplotlib.pyplot as plt
import seaborn as sns


latent_dim = 1
X = np.random.normal(size=(10**7, 1))
z = np.random.uniform(-1, 1, (10**7, latent_dim))

inputs = Input(shape=(1,))
x = Dense(32)(inputs)
x = BatchNormalization(axis=-1)(x)
x = Activation("relu")(x)

x = Dense(32)(x)
x = BatchNormalization(axis=-1)(x)
x = Activation("relu")(x)

x = Dense(32)(x)
x = BatchNormalization(axis=-1)(x)
x = Activation("relu")(x)

x = Dense(32)(x)
x = BatchNormalization(axis=-1)(x)
x = Activation("relu")(x)

x = Dense(1)(x)
model = Model(input=inputs, output=x)


def loss(y_true, y_pred):
    l = K.mean(K.square(K.mean(y_true, axis=0) - K.mean(y_pred, axis=0)), axis=-1)
    l += K.mean(K.abs(K.var(y_true, axis=0) - K.var(y_pred, axis=0)), axis=-1)
    return l

model.compile(optimizer="adam", loss=loss)
model.fit(z, X, batch_size=256, nb_epoch=10, verbose=1, shuffle=True)

z = np.random.uniform(-1, 1, (10 ** 5, latent_dim))
preds = model.predict(z, batch_size=256)

fig = plt.figure()
plt.xlim(xmin=-6, xmax=6)
plt.ylim(ymin=0., ymax=0.5)
sns.kdeplot(X.reshape(-1), label="x")
fig.hold(True)
sns.kdeplot(preds.reshape(-1), label="g(z)")
fig.savefig("result.png")
