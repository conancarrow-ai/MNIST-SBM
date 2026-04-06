"""Training loop for the sparse Boltzmann machine."""

import json
import os
import sys
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

sys.path.insert(0, "jax_fid")
from jax_fid.fid import compute_statistics, compute_frechet_distance, get_inception_apply_fn

from config import TrainConfig
from load_mnist_data import load_mnist
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

    # --- FID setup ---
    print("Loading InceptionV3 for FID...", flush=True)
    fid_feature_fn = get_inception_apply_fn()
    fid_stats = np.load(cfg.fid_stats_path)
    mu_real, sigma_real = fid_stats["mu"], fid_stats["sigma"]

    print(f"\nTraining: epochs {start_epoch+1}..{cfg.n_epochs}, {n_batches} batches, "
          f"batch_size={cfg.batch_size}", flush=True)
    print(f"  lr={cfg.learning_rate}, momentum={cfg.momentum_alpha}, beta={cfg.beta}", flush=True)
    print(f"  schedule=({cfg.sched_warmup}, {cfg.sched_n_samples}, "
          f"{cfg.sched_steps_per_sample})\n", flush=True)

    log_path = os.path.join(cfg.checkpoint_dir, "train_log.jsonl")
    log_file = open(log_path, "a")

    if cfg.double_fid_warmup:
        warmup_model = IsingEBM(nodes, edges, biases, weights, beta_arr)
        for w_round in range(1, 3):
            print(f"FID warmup {w_round}/2...", flush=True)
            key, k_fid_warmup = jax.random.split(key)
            t_fw0 = time.time()
            fid_warmup_samples = generate(k_fid_warmup, warmup_model, z1, schedule, cfg.fid_n_samples)
            fid_warmup_samples.block_until_ready()
            t_fw_gen = time.time() - t_fw0
            t_fw1 = time.time()
            fid_warmup_images = np.array(fid_warmup_samples, dtype=np.float32).reshape(-1, 28, 28, 1)
            mu_gen_w, sigma_gen_w = compute_statistics(fid_warmup_images, fid_feature_fn, 100)
            fid_w, _, _ = compute_frechet_distance(mu_real, mu_gen_w, sigma_real, sigma_gen_w)
            t_fw_inc = time.time() - t_fw1
            print(f"  fid_gen: {t_fw_gen:.1f}s — fid_inception: {t_fw_inc:.1f}s — FID: {fid_w:.2f}",
                  flush=True)

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

            print_interval = max(1, n_batches // 5)
            if (b + 1) % print_interval == 0 or b == n_batches - 1:
                print(f"  epoch {epoch+1} batch {b+1}/{n_batches} — {time.time()-tb:.2f}s — "
                      f"|grad_w|={float(jnp.abs(grad_w).mean()):.6f} "
                      f"|grad_b|={float(jnp.abs(grad_b).mean()):.6f} "
                      f"|w|={float(jnp.abs(weights).mean()):.6f} "
                      f"|b|={float(jnp.abs(biases).mean()):.6f}",
                      flush=True)

        t_train = time.time() - t0
        key, k_gen, k_fid = jax.random.split(key, 3)
        model = IsingEBM(nodes, edges, biases, weights, beta_arr)

        t_gen0 = time.time()
        imgs = generate(k_gen, model, z1, schedule, cfg.n_gen)
        act = float(imgs.mean())
        t_gen = time.time() - t_gen0

        # FID
        t_fid0 = time.time()
        fid_samples = generate(k_fid, model, z1, schedule, cfg.fid_n_samples)
        fid_samples.block_until_ready()
        t_fid_gen = time.time() - t_fid0

        t_fid_inc0 = time.time()
        fid_images = np.array(fid_samples, dtype=np.float32).reshape(-1, 28, 28, 1)
        mu_gen, sigma_gen = compute_statistics(fid_images, fid_feature_fn, 100)
        fid_score, _, _ = compute_frechet_distance(mu_real, mu_gen, sigma_real, sigma_gen)
        t_fid_inc = time.time() - t_fid_inc0

        print(f"Epoch {epoch+1}/{cfg.n_epochs} — train: {t_train:.1f}s — gen: {t_gen:.1f}s — "
              f"fid_gen: {t_fid_gen:.1f}s — fid_inception: {t_fid_inc:.1f}s — "
              f"act: {act:.3f} — FID: {fid_score:.2f}",
              flush=True)

        log_entry = {
            "epoch": epoch + 1,
            "time_train": round(t_train, 2),
            "time_gen": round(t_gen, 2),
            "time_fid_gen": round(t_fid_gen, 2),
            "time_fid_inception": round(t_fid_inc, 2),
            "time_per_batch": round(t_train / n_batches, 4),
            "mean_activation": round(act, 4),
            "fid": round(float(fid_score), 2),
            "lr": cfg.learning_rate,
            "momentum": cfg.momentum_alpha,
            "beta": cfg.beta,
            "batch_size": cfg.batch_size,
            "sched_warmup": cfg.sched_warmup,
            "sched_n_samples": cfg.sched_n_samples,
            "sched_steps_per_sample": cfg.sched_steps_per_sample,
            "fid_n_samples": cfg.fid_n_samples,
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

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

    log_file.close()
    print(f"Training log: {log_path}", flush=True)

    return weights, biases, z1
