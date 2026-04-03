"""Training configuration for the sparse Boltzmann machine."""


class TrainConfig:
    def __init__(
        self,
        # --- Data ---
        data_dir: str = "data/MNIST/raw",         # path to raw MNIST IDX files
        # --- Graph ---
        grid_size: int = 60,                      # side length of the square grid
        connections: tuple = ((1,0),(2,1),(2,3),(4,1)),  # knight's-move-style wiring
        # --- Training ---
        n_epochs: int = 100,
        batch_size: int = 50,
        learning_rate: float = 0.003,
        momentum_alpha: float = 0.6,
        beta: float = 1.0,
        max_batches: int | None = None,           # cap batches per epoch (None = all)
        seed: int = 42,
        # --- Sampling schedule ---
        sched_warmup: int = 5,
        sched_n_samples: int = 20,
        sched_steps_per_sample: int = 5,
        # --- Generation / checkpointing ---
        gen_every: int = 10,                      # save generated samples every N epochs
        n_gen: int = 10,                          # number of images to generate
        checkpoint_dir: str = "checkpoints",
        checkpoint_every: int = 5,                # save checkpoint every N epochs
        resume_from: str | None = None,           # path to .npz checkpoint to resume from
    ):
        self.data_dir = data_dir
        self.grid_size = grid_size
        self.connections = connections
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum_alpha = momentum_alpha
        self.beta = beta
        self.max_batches = max_batches
        self.seed = seed
        self.sched_warmup = sched_warmup
        self.sched_n_samples = sched_n_samples
        self.sched_steps_per_sample = sched_steps_per_sample
        self.gen_every = gen_every
        self.n_gen = n_gen
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.resume_from = resume_from
