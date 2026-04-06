"""Compute and cache InceptionV3 statistics for real MNIST training data."""

import sys

import numpy as np

sys.path.insert(0, "jax_fid")
from jax_fid.fid import compute_statistics, get_inception_apply_fn

from load_mnist_data import load_mnist

FID_BATCH_SIZE = 100

print("Loading InceptionV3...", flush=True)
feature_fn = get_inception_apply_fn()

print("Loading MNIST...", flush=True)
train_img, _ = load_mnist()

print("Computing real MNIST statistics...", flush=True)
real_images = np.array(train_img, dtype=np.float32).reshape(-1, 28, 28, 1)
mu, sigma = compute_statistics(real_images, feature_fn, FID_BATCH_SIZE)

np.savez("mnist_inception_stats.npz", mu=np.array(mu), sigma=np.array(sigma))
print("Saved mnist_inception_stats.npz", flush=True)
