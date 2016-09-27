from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
from six import u

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def get_chars(char_filepath):
    chars = []
    with open(char_filepath, "r") as f:
        for line in f:
            c = line.rstrip().decode("utf-8")
            chars.append(c)
    return chars


def gen_char(c, font):
    image = Image.new("L", (args.size * 2, args.size * 2), color=0)
    draw = ImageDraw.Draw(image)

    draw.text((args.size / 2, args.size / 2), c, fill=255, font=font)
    del draw

    image = np.asarray(image, dtype=np.float32) / 255.
    x_flat = np.sum(image, axis=0)
    y_flat = np.sum(image, axis=1)

    x_flag = False
    y_flag = False
    x_range = [0, 0]
    y_range = [0, 0]
    for i in range(x_flat.shape[0]):
        if x_flat[i] > 0 and not x_flag:
            x_range[0] = i
            x_flag = True
        elif x_flat[i] == 0 and x_flag:
            x_range[1] = i
            x_flag = False

        if y_flat[i] > 0 and not y_flag:
            y_range[0] = i
            y_flag = True
        elif y_flat[i] == 0 and y_flag:
            y_range[1] = i
            y_flag = False

    w = x_range[1] - x_range[0]
    h = y_range[1] - y_range[0]

    a = max(w, h)
    x_mean = sum(x_range) / 2
    y_mean = sum(y_range) / 2
    x_range = [x_mean - a / 2, x_mean + a / 2]
    y_range = [y_mean - a / 2, y_mean + a / 2]

    image = image[y_range[0] - 1: y_range[1] + 1, x_range[0] - 1: x_range[1] + 1]

    return image


def run(args):
    font_paths = args.fonts.split(",")
    chars = get_chars(args.char_filepath)
    fonts = [ImageFont.truetype(font_path, size=args.size) for font_path in font_paths]

    for c in chars:
        im = gen_char(c, fonts[0])
        print(im.shape)
        plt.imshow(im, cmap=plt.get_cmap("gray"))
        plt.savefig("%s.png" % c)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fonts", dest="fonts", type=str)
    parser.add_argument("--char_file", dest="char_filepath", type=str)
    parser.add_argument("--size", dest="size", type=int, default=12)
    args = parser.parse_args()

    run(args)
