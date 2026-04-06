"""Training configuration for the sparse Boltzmann machine."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # --- Data ---
    data_dir: str = "data/MNIST/raw"              # path to raw MNIST IDX files
    # --- Graph ---
    grid_size: int = 60                           # side length of the square grid
    connections: tuple = ((1,0),(2,1),(2,3),(4,1)) # knight's-move-style wiring
    # --- Training ---
    n_epochs: int = 100
    batch_size: int = 50
    learning_rate: float = 0.003
    momentum_alpha: float = 0.6
    beta: float = 1.0
    max_batches: int | None = None                # cap batches per epoch (None = all)
    seed: int = 42
    print_every: int = 1
    # --- Sampling schedule ---
    sched_warmup: int = 5
    sched_n_samples: int = 20
    sched_steps_per_sample: int = 5
    # --- Generation / checkpointing ---
    gen_every: int = 1                            # save generated samples every N epochs
    n_gen: int = 10                               # number of images to generate
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 1                     # save checkpoint every N epochs
    resume_from: str | None = None                # path to .npz checkpoint to resume from
    # --- FID ---
    fid_n_samples: int = 100                      # samples to generate for FID each epoch
    fid_stats_path: str = "mnist_inception_stats.npz"  # cached real MNIST statistics
    double_fid_warmup: bool = False               # run two FID evals before training (2nd gives true timing)
