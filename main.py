"""Training a sparse Boltzmann machine on MNIST using thrml on a 60x60 grid graph."""

from config import TrainConfig

# ── Edit this to configure your run ──────────────────────────────────────────
cfg = TrainConfig(
    n_epochs=20,
    batch_size=50,
    learning_rate=0.003,
    # total steps per chain = sched_warmup + (sched_n_samples - 1) * sched_steps_per_sample
    sched_warmup=10,
    sched_n_samples=100,
    sched_steps_per_sample=10,
)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from train import train
    train(cfg)
