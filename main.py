"""Training a sparse Boltzmann machine on MNIST using thrml on a 60x60 grid graph."""

from config import TrainConfig

# ── Edit this to configure your run ──────────────────────────────────────────
cfg = TrainConfig(
    n_epochs=50,
    batch_size=200,
    learning_rate=0.03,
    momentum_alpha=0.6,
    # total steps per chain = sched_warmup + (sched_n_samples - 1) * sched_steps_per_sample
    sched_warmup=1000,
    sched_n_samples=100,
    sched_steps_per_sample=20,
    fid_n_samples=5000,
    double_fid_warmup=False
)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from train import train
    train(cfg)
