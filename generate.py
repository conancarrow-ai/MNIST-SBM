"""Generate MNIST digits via annealed sampling from a trained Ising model."""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from graph import Z1Graph, color_blocks
from thrml.block_management import Block, block_state_to_global, from_global_state
from thrml.block_sampling import _run_blocks
from thrml.conditional_samplers import AbstractConditionalSampler
from thrml.models.ising import IsingEBM, IsingSamplingProgram, hinton_init

# Load checkpoint
d = np.load("./epoch_0043.npz")
weights = jnp.array(d["weights"])
biases = jnp.array(d["biases"])

# Rebuild graph (same seed as training)
np.random.seed(42)
z1 = Z1Graph()
nodes, edges, pixel_nodes = z1.node_list, z1.edge_list, z1.data_nodes
free_blocks = color_blocks(z1.coloring, z1.n_colors, nodes)
pixel_block = Block(pixel_nodes)

# Annealed generation
n_images = 10
beta_step = 0.0325
steps_per_beta = 400
betas = np.arange(0, 2.0, beta_step)

key = jax.random.key(2)

# Initialize at beta=0 (random)
model0 = IsingEBM(nodes, edges, biases, weights, jnp.array(0.0))
key, k_init = jax.random.split(key)
state = hinton_init(k_init, model0, free_blocks, (n_images,))

for i, beta in enumerate(betas):
    model = IsingEBM(nodes, edges, biases, weights, jnp.array(float(beta)))
    program = IsingSamplingProgram(model, free_blocks, [])

    sampler_states = jax.tree.map(
        lambda x: x.init(),
        program.samplers,
        is_leaf=lambda a: isinstance(a, AbstractConditionalSampler),
    )

    key, k_run = jax.random.split(key)
    # _run_blocks returns (final_chain_state, final_sampler_states)
    state, _ = jax.vmap(
        lambda k, s: _run_blocks(k, program, s, [], steps_per_beta, sampler_states)
    )(jax.random.split(k_run, n_images), state)

    if (i + 1) % 10 == 0:
        print(f"  beta={float(beta):.3f}", flush=True)

# Extract pixel values from final block state
model_final = IsingEBM(nodes, edges, biases, weights, jnp.array(5.0))
program_final = IsingSamplingProgram(model_final, free_blocks, [])

def extract_pixels(s):
    gs = block_state_to_global(s, program_final.gibbs_spec)
    return from_global_state(gs, program_final.gibbs_spec, [pixel_block])

pixels = jax.vmap(extract_pixels)(state)

# Display
fig, axes = plt.subplots(1, n_images, figsize=(2 * n_images, 2))
for i, ax in enumerate(axes):
    img = np.array(pixels[0][i]).reshape(28, 28)
    ax.imshow(img, cmap="gray")
    ax.axis("off")
fig.suptitle("Annealed samples (beta 0 -> 5, step 0.125)")
plt.tight_layout()
plt.savefig("Generated_MNIST_Digits.png", dpi=150, bbox_inches="tight")
print("Saved Generated_MNIST_Digits.png", flush=True)
plt.close(fig)
