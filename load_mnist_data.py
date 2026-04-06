"""MNIST data loading."""

import os
import struct

import numpy as np


def load_mnist(data_dir="data/MNIST/raw"):
    def read_images(path):
        with open(path, "rb") as f:
            _, n, rows, cols = struct.unpack(">4I", f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)

    train = (read_images(os.path.join(data_dir, "train-images-idx3-ubyte")) >= 128)
    test = (read_images(os.path.join(data_dir, "t10k-images-idx3-ubyte")) >= 128)
    return train, test
