"""Model utilities: bias init, generation, and checkpoint I/O."""

import jax
import jax.numpy as jnp
import numpy as np

from thrml.block_management import Block
from thrml.block_sampling import SamplingSchedule, sample_states
from thrml.models.ising import (
    IsingEBM,
    IsingSamplingProgram,
    hinton_init,
)

from graph import color_blocks


def init_biases(nodes, pixel_nodes, train_images):
    node_idx = {n: i for i, n in enumerate(nodes)}
    biases = np.zeros(len(nodes), dtype=np.float32)
    eps = 1e-6
    p = np.clip(train_images.mean(axis=0), eps, 1 - eps)
    for i, n in enumerate(pixel_nodes):
        biases[node_idx[n]] = np.log(p[i] / (1 - p[i]))
    return jnp.array(biases)


def generate(key, model, z1, schedule, n_images=10):
    free_blocks = color_blocks(z1.coloring, z1.n_colors, z1.node_list)

    program = IsingSamplingProgram(model, free_blocks, [])
    key, k_init = jax.random.split(key)
    init_state = hinton_init(k_init, model, free_blocks, (n_images,))
    keys = jax.random.split(key, n_images)

    pixel_block = Block(z1.data_nodes)
    samples = jax.vmap(
        lambda k, s: sample_states(k, program, schedule, s, [], [pixel_block])
    )(keys, init_state)

    return samples[0][:, -1, :]


def save_checkpoint(path, epoch, weights, biases, vel_w, vel_b):
    np.savez(path,
             epoch=epoch,
             weights=np.array(weights),
             biases=np.array(biases),
             vel_w=np.array(vel_w),
             vel_b=np.array(vel_b))


def load_checkpoint(path):
    d = np.load(path)
    return (int(d["epoch"]),
            jnp.array(d["weights"]),
            jnp.array(d["biases"]),
            jnp.array(d["vel_w"]),
            jnp.array(d["vel_b"]))
