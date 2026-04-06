"""Benchmark grad_step wall-clock time across batch sizes."""

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from thrml.block_management import Block
from thrml.block_sampling import SamplingSchedule
from thrml.models.ising import (
    IsingEBM,
    IsingTrainingSpec,
    estimate_kl_grad,
    hinton_init,
)

from data import load_mnist
from graph import Z1Graph, color_blocks
from model import init_biases

# ── Configuration ────────────────────────────────────────────────────────────
BATCH_SIZES = [50, 100, 200, 400, 800, 1600, 3200, 6400]
SCHED_WARMUP = 1000
SCHED_N_SAMPLES = 30
SCHED_STEPS_PER_SAMPLE = 30
BETA = 1.0
SEED = 42
N_REPS = 3              # repetitions per batch size (after warmup)
# ─────────────────────────────────────────────────────────────────────────────

print("Loading MNIST...", flush=True)
train_img, _ = load_mnist()

print("Building Z1 graph...", flush=True)
np.random.seed(SEED)
z1 = Z1Graph()
nodes, edges, pixel_nodes = z1.node_list, z1.edge_list, z1.data_nodes

free_blocks = color_blocks(z1.coloring, z1.n_colors, nodes)
clamped_blocks = color_blocks(z1.coloring, z1.n_colors, nodes, exclude_nodes=pixel_nodes)

key = jax.random.key(SEED)
key, k_w = jax.random.split(key)
weights = 0.01 * jax.random.normal(k_w, (len(edges),))
biases = init_biases(nodes, pixel_nodes, train_img)
beta_arr = jnp.array(BETA)

schedule = SamplingSchedule(SCHED_WARMUP, SCHED_N_SAMPLES, SCHED_STEPS_PER_SAMPLE)
data_blocks = [Block(pixel_nodes)]

print(f"\nBenchmarking grad_step for batch sizes: {BATCH_SIZES}")
print(f"  schedule=({SCHED_WARMUP}, {SCHED_N_SAMPLES}, {SCHED_STEPS_PER_SAMPLE})")
print(f"  {N_REPS} reps per batch size (after 1 warmup)\n", flush=True)

for bs in BATCH_SIZES:
    @eqx.filter_jit
    def grad_step(weights, biases, key, batch_data):
        model = IsingEBM(nodes, edges, biases, weights, beta_arr)
        spec = IsingTrainingSpec(model, data_blocks, [], clamped_blocks, free_blocks, schedule, schedule)
        k_pos, k_neg, k_grad = jax.random.split(key, 3)
        init_pos = hinton_init(k_pos, model, clamped_blocks, (1, bs))
        init_neg = hinton_init(k_neg, model, free_blocks, (bs,))
        gw, gb, _, _ = estimate_kl_grad(
            k_grad, spec, nodes, edges, [batch_data], [], init_pos, init_neg
        )
        return gw, gb

    idx = np.random.choice(len(train_img), bs, replace=False)
    batch_data = jnp.array(train_img[idx], dtype=jnp.bool_)

    # Warmup (includes JIT compilation)
    key, k = jax.random.split(key)
    gw, gb = grad_step(weights, biases, k, batch_data)
    gw.block_until_ready()

    # Timed reps
    times = []
    for r in range(N_REPS):
        key, k = jax.random.split(key)
        t0 = time.time()
        gw, gb = grad_step(weights, biases, k, batch_data)
        gw.block_until_ready()
        times.append(time.time() - t0)

    avg = sum(times) / len(times)
    print(f"  batch_size={bs:>4d}  avg={avg:.3f}s  per_sample={avg/bs:.5f}s  runs={times}", flush=True)
