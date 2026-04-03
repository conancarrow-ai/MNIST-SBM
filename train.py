"""Training loop for the sparse Boltzmann machine."""

import os
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

from config import TrainConfig
from data import load_mnist
from graph import Z1Graph, color_blocks
from model import init_biases, generate, save_checkpoint, load_checkpoint

def train(cfg: TrainConfig):
    key = jax.random.key(cfg.seed)

    print("Loading MNIST...", flush=True)
    train_img, _ = load_mnist(cfg.data_dir)
    n_train = len(train_img)
    n_batches = n_train // cfg.batch_size
    if cfg.max_batches is not None:
        n_batches = min(n_batches, cfg.max_batches)

    print("Building Z1 graph...", flush=True)
    np.random.seed(cfg.seed)
    z1 = Z1Graph(grid_size=cfg.grid_size, connections=cfg.connections)
    nodes, edges, pixel_nodes = z1.node_list, z1.edge_list, z1.data_nodes
    print(f"  {len(nodes)} nodes, {len(edges)} edges, {len(pixel_nodes)} visible, "
          f"{len(nodes) - len(pixel_nodes)} hidden", flush=True)

    free_blocks = color_blocks(z1.coloring, z1.n_colors, nodes)
    clamped_blocks = color_blocks(z1.coloring, z1.n_colors, nodes, exclude_nodes=pixel_nodes)

    key, k_w = jax.random.split(key)
    weights = 0.01 * jax.random.normal(k_w, (len(edges),))
    biases = init_biases(nodes, pixel_nodes, train_img)
    beta_arr = jnp.array(cfg.beta)

    schedule = SamplingSchedule(cfg.sched_warmup, cfg.sched_n_samples, cfg.sched_steps_per_sample)

    vel_w = jnp.zeros_like(weights)
    vel_b = jnp.zeros_like(biases)
    start_epoch = 0

    if cfg.resume_from is not None:
        start_epoch, weights, biases, vel_w, vel_b = load_checkpoint(cfg.resume_from)
        print(f"Resumed from {cfg.resume_from} at epoch {start_epoch}", flush=True)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    data_blocks = [Block(pixel_nodes)]

    @eqx.filter_jit
    def grad_step(weights, biases, key, batch_data):
        model = IsingEBM(nodes, edges, biases, weights, beta_arr)
        spec = IsingTrainingSpec(model, data_blocks, [], clamped_blocks, free_blocks, schedule, schedule)
        k_pos, k_neg, k_grad = jax.random.split(key, 3)
        init_pos = hinton_init(k_pos, model, clamped_blocks, (1, cfg.batch_size))
        init_neg = hinton_init(k_neg, model, free_blocks, (cfg.batch_size,))
        gw, gb, _, _ = estimate_kl_grad(
            k_grad, spec, nodes, edges, [batch_data], [], init_pos, init_neg
        )
        return gw, gb

    print(f"\nTraining: epochs {start_epoch+1}..{cfg.n_epochs}, {n_batches} batches, "
          f"batch_size={cfg.batch_size}", flush=True)
    print(f"  lr={cfg.learning_rate}, momentum={cfg.momentum_alpha}, beta={cfg.beta}", flush=True)
    print(f"  schedule=({cfg.sched_warmup}, {cfg.sched_n_samples}, "
          f"{cfg.sched_steps_per_sample})\n", flush=True)

    rng = np.random.RandomState(cfg.seed)
    for _ in range(start_epoch):
        rng.permutation(n_train)

    for epoch in range(start_epoch, cfg.n_epochs):
        t0 = time.time()
        perm = rng.permutation(n_train)

        for b in range(n_batches):
            tb = time.time()
            idx = perm[b * cfg.batch_size : (b + 1) * cfg.batch_size]
            batch_data = jnp.array(train_img[idx], dtype=jnp.bool_)

            key, k_step = jax.random.split(key)
            grad_w, grad_b = grad_step(weights, biases, k_step, batch_data)

            vel_w = -(cfg.learning_rate / cfg.beta) * grad_w + cfg.momentum_alpha * vel_w
            vel_b = -(cfg.learning_rate / cfg.beta) * grad_b + cfg.momentum_alpha * vel_b
            weights = weights + vel_w
            biases = biases + vel_b

            if (b + 1) % 20 == 0:
                print(f"  epoch {epoch+1} batch {b+1}/{n_batches} — {time.time()-tb:.2f}s",
                      flush=True)

        dt = time.time() - t0
        key, k_gen = jax.random.split(key)
        model = IsingEBM(nodes, edges, biases, weights, beta_arr)
        imgs = generate(k_gen, model, z1, schedule, cfg.n_gen)
        act = float(imgs.mean())
        print(f"Epoch {epoch+1}/{cfg.n_epochs} — {dt:.1f}s — mean activation: {act:.3f}",
              flush=True)

        if cfg.gen_every and (epoch + 1) % cfg.gen_every == 0:
            img_path = os.path.join(cfg.checkpoint_dir, f"samples_epoch_{epoch+1:04d}.npy")
            np.save(img_path, np.array(imgs))
            print(f"  saved {img_path}", flush=True)

        if cfg.checkpoint_every and (epoch + 1) % cfg.checkpoint_every == 0:
            cp = os.path.join(cfg.checkpoint_dir, f"epoch_{epoch+1:04d}.npz")
            save_checkpoint(cp, epoch + 1, weights, biases, vel_w, vel_b)
            print(f"  saved {cp}", flush=True)

    final = os.path.join(cfg.checkpoint_dir, "final.npz")
    save_checkpoint(final, cfg.n_epochs, weights, biases, vel_w, vel_b)
    print(f"Saved final model to {final}", flush=True)

    return weights, biases, z1
