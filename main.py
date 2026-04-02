"""Training a sparse Boltzmann machine on MNIST using thrml on a 60x60 grid graph."""

import os
import struct
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx

from thrml.block_management import Block
from thrml.block_sampling import SamplingSchedule, sample_states
from thrml.models.ising import (
    IsingEBM,
    IsingSamplingProgram,
    IsingTrainingSpec,
    estimate_kl_grad,
    hinton_init,
)
from thrml.pgm import SpinNode


def load_mnist(data_dir="data/MNIST/raw"):
    def read_images(path):
        with open(path, "rb") as f:
            _, n, rows, cols = struct.unpack(">4I", f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)

    train = (read_images(os.path.join(data_dir, "train-images-idx3-ubyte")) >= 128)
    test = (read_images(os.path.join(data_dir, "t10k-images-idx3-ubyte")) >= 128)
    return train, test


class Z1Graph:
    """60x60 grid graph with extra knight's-move-style connections."""

    def __init__(self, grid_size=60, data_size=784, connections=((1,0),(2,1),(2,3),(4,1))):
        self.grid_size = grid_size
        self.graph = nx.grid_graph(dim=(grid_size, grid_size), periodic=False)

        self.coord_to_node = {coord: SpinNode() for coord in self.graph.nodes}
        nx.relabel_nodes(self.graph, self.coord_to_node, copy=False)
        for node, coord in ((v, k) for k, v in self.coord_to_node.items()):
            self.graph.nodes[node]["coords"] = coord

        self.coloring = nx.bipartite.color(self.graph)
        self.color_0 = [n for n, c in self.coloring.items() if c == 0]
        self.color_1 = [n for n, c in self.coloring.items() if c == 1]
        self.n_colors = max(self.coloring.values()) + 1

        for c in connections:
            self._wire(c)

        self.node_list = list(self.graph.nodes)
        self.edge_list = list(self.graph.edges)

        self.data_node_indices = np.random.choice(len(self.color_0), data_size, replace=False)
        self.data_nodes = [self.color_0[x] for x in self.data_node_indices]

    def _wire(self, c):
        a, b = c
        for n in self.graph:
            x, y = self.graph.nodes[n]["coords"]
            for m in [(x+a, y+b), (x-b, y+a), (x-a, y-b), (x+b, y-a)]:
                if 0 <= m[0] < self.grid_size and 0 <= m[1] < self.grid_size:
                    self.graph.add_edge(n, self.coord_to_node[m])


def color_blocks(coloring, n_colors, nodes, exclude_nodes=None):
    exclude = set(exclude_nodes) if exclude_nodes else set()
    groups = [[] for _ in range(n_colors)]
    for node in nodes:
        if node not in exclude:
            groups[coloring[node]].append(node)
    return [Block(g) for g in groups if g]


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


def train(
    n_epochs=100,
    batch_size=50,
    learning_rate=0.003,
    momentum_alpha=0.6,
    beta=1.0,
    grid_size=60,
    connections=((1,0),(2,1),(2,3),(4,1)),
    sched_warmup=5,
    sched_n_samples=20,
    sched_steps_per_sample=5,
    gen_every=10,
    n_gen=10,
    max_batches=None,
    seed=42,
    checkpoint_dir="checkpoints",
    checkpoint_every=5,
    resume_from=None,
):
    key = jax.random.key(seed)

    print("Loading MNIST...")
    train_img, _ = load_mnist()
    n_train = len(train_img)
    n_batches = n_train // batch_size
    if max_batches is not None:
        n_batches = min(n_batches, max_batches)

    print("Building Z1 graph...")
    np.random.seed(seed)
    z1 = Z1Graph(grid_size=grid_size, connections=connections)
    nodes, edges, pixel_nodes = z1.node_list, z1.edge_list, z1.data_nodes
    print(f"  {len(nodes)} nodes, {len(edges)} edges, {len(pixel_nodes)} visible, "
          f"{len(nodes) - len(pixel_nodes)} hidden")

    free_blocks = color_blocks(z1.coloring, z1.n_colors, nodes)
    clamped_blocks = color_blocks(z1.coloring, z1.n_colors, nodes, exclude_nodes=pixel_nodes)

    key, k_w = jax.random.split(key)
    weights = 0.01 * jax.random.normal(k_w, (len(edges),))
    biases = init_biases(nodes, pixel_nodes, train_img)
    beta_arr = jnp.array(beta)

    schedule = SamplingSchedule(sched_warmup, sched_n_samples, sched_steps_per_sample)

    vel_w = jnp.zeros_like(weights)
    vel_b = jnp.zeros_like(biases)
    start_epoch = 0

    if resume_from is not None:
        start_epoch, weights, biases, vel_w, vel_b = load_checkpoint(resume_from)
        print(f"Resumed from {resume_from} at epoch {start_epoch}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    data_blocks = [Block(pixel_nodes)]

    @eqx.filter_jit
    def grad_step(weights, biases, key, batch_data):
        model = IsingEBM(nodes, edges, biases, weights, beta_arr)
        spec = IsingTrainingSpec(model, data_blocks, [], clamped_blocks, free_blocks, schedule, schedule)
        k_pos, k_neg, k_grad = jax.random.split(key, 3)
        init_pos = hinton_init(k_pos, model, clamped_blocks, (1, batch_size))
        init_neg = hinton_init(k_neg, model, free_blocks, (batch_size,))
        gw, gb, _, _ = estimate_kl_grad(
            k_grad, spec, nodes, edges, [batch_data], [], init_pos, init_neg
        )
        return gw, gb

    print(f"\nTraining: epochs {start_epoch+1}..{n_epochs}, {n_batches} batches, batch_size={batch_size}")
    print(f"  lr={learning_rate}, momentum={momentum_alpha}, beta={beta}")
    print(f"  schedule=({sched_warmup}, {sched_n_samples}, {sched_steps_per_sample})\n")

    rng = np.random.RandomState(seed)
    for _ in range(start_epoch):
        rng.permutation(n_train)

    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()
        perm = rng.permutation(n_train)

        for b in range(n_batches):
            tb = time.time()
            idx = perm[b * batch_size : (b + 1) * batch_size]
            batch_data = jnp.array(train_img[idx], dtype=jnp.bool_)

            key, k_step = jax.random.split(key)
            grad_w, grad_b = grad_step(weights, biases, k_step, batch_data)

            vel_w = -(learning_rate / beta) * grad_w + momentum_alpha * vel_w
            vel_b = -(learning_rate / beta) * grad_b + momentum_alpha * vel_b
            weights = weights + vel_w
            biases = biases + vel_b

            if (b + 1) % 20 == 0:
                print(f"  epoch {epoch+1} batch {b+1}/{n_batches} — {time.time()-tb:.2f}s")

        dt = time.time() - t0
        key, k_gen = jax.random.split(key)
        model = IsingEBM(nodes, edges, biases, weights, beta_arr)
        imgs = generate(k_gen, model, z1, schedule, n_gen)
        act = float(imgs.mean())
        print(f"Epoch {epoch+1}/{n_epochs} — {dt:.1f}s — mean activation: {act:.3f}")

        if gen_every and (epoch + 1) % gen_every == 0:
            img_path = os.path.join(checkpoint_dir, f"samples_epoch_{epoch+1:04d}.npy")
            np.save(img_path, np.array(imgs))
            print(f"  saved {img_path}")

        if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
            cp = os.path.join(checkpoint_dir, f"epoch_{epoch+1:04d}.npz")
            save_checkpoint(cp, epoch + 1, weights, biases, vel_w, vel_b)
            print(f"  saved {cp}")

    final = os.path.join(checkpoint_dir, "final.npz")
    save_checkpoint(final, n_epochs, weights, biases, vel_w, vel_b)
    print(f"Saved final model to {final}")

    return weights, biases, z1


if __name__ == "__main__":
    train()
