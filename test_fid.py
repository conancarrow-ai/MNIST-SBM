"""Quick FID test: 5 samples from final checkpoint only."""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "jax_fid")
from jax_fid.fid import compute_statistics, compute_frechet_distance, get_inception_apply_fn

from graph import Z1Graph
from model import generate
from thrml.block_sampling import SamplingSchedule
from thrml.models.ising import IsingEBM

np.random.seed(42)
z1 = Z1Graph()
nodes, edges = z1.node_list, z1.edge_list
schedule = SamplingSchedule(200, 30, 50)

d = np.load("checkpoints_runpod/final.npz")
model = IsingEBM(nodes, edges, jnp.array(d["biases"]), jnp.array(d["weights"]), jnp.array(1.0))

print("Generating 5 samples...", flush=True)
t0 = time.time()
key = jax.random.key(0)
samples = generate(key, model, z1, schedule, n_images=5)
samples.block_until_ready()
print(f"  done in {time.time() - t0:.1f}s", flush=True)

print("Loading InceptionV3...", flush=True)
feature_fn = get_inception_apply_fn()

print("Loading cached MNIST stats...", flush=True)
stats = np.load("mnist_inception_stats.npz")
mu_real, sigma_real = stats["mu"], stats["sigma"]

gen_images = np.array(samples, dtype=np.float32).reshape(-1, 28, 28, 1)
mu_gen, sigma_gen = compute_statistics(gen_images, feature_fn, 5)

fid_score, _, _ = compute_frechet_distance(mu_real, mu_gen, sigma_real, sigma_gen)
print(f"FID (5 samples, final checkpoint) = {fid_score:.2f}", flush=True)
