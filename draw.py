"""Extract NxN patch from an image using DRAW.

image: (batch_size, channel, height, width)
gX, gY: (batch_size, 1)
deltaX, deltaY: (batch_size, 1)
muX, muY: (batch_size, N)
sigmaX, sigmaY: (batch_size, 1)
FX, FY: (batch_size, N, A), (batch_size, N, B)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.misc

import theano
import theano.tensor as T

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

image = scipy.misc.face()
image = image.transpose([2, 0, 1])
image = np.expand_dims(image, 0)
image = image.astype("float32") / 255.

width = image.shape[3]
height = image.shape[2]

N = 128  # extract NxN patch

gX = np.asarray([width / 2])[:, None]
gY = np.asarray([height / 2])[:, None]

deltaX = np.asarray([1])[:, None]
deltaY = np.asarray([1])[:, None]

i = np.arange(N)
j = np.arange(N)
a = np.arange(width)
b = np.arange(height)

muX = gX + (i - N / 2 - 0.5) * deltaX
muY = gY + (j - N / 2 - 0.5) * deltaY

sigmaX = np.asarray([1.0])[:, None]
sigmaY = np.asarray([1.0])[:, None]

FX = np.exp(-np.square(a - muX[:, :, None]) / (2 * np.square(sigmaX)))
FX /= np.sum(FX, axis=-1)[:, :, None]
FY = np.exp(-np.square(b - muY[:, :, None]) / (2 * np.square(sigmaY)))
FY /= np.sum(FY, axis=-1)[:, :, None]

FYx = (FY[:, None, :, :, None] * image[:, :, None, :, :]).sum(axis=3)
FXT = FX.transpose([0, 2, 1])
FYxFXT = (FYx[:, :, :, :, None] * FXT[:, None, None, :, :]).sum(axis=3)

FYT = FY.transpose([0, 2, 1])
FYTx = (FYT[:, None, :, :, None] * FYxFXT[:, :, None, :, :]).sum(axis=3)
recon = (FYTx[:, :, :, :, None] * FX[:, None, None, :, :]).sum(axis=3)

plt.subplot(311)
plt.imshow(image[0].transpose([1, 2, 0]))
plt.subplot(312)
plt.imshow(FYxFXT[0].transpose([1, 2, 0]))
plt.subplot(313)
plt.imshow(recon[0].transpose([1, 2, 0]))
plt.show()
