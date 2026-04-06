# MNIST-SBM

Train a sparse Boltzmann machine on MNIST using [thrml](https://github.com/extropic-ai/thrml) on a 60x60 grid graph with knight's-move-style wiring.

## Setup

Requires Python 3.10-3.13.

```bash
git clone git@github.com:extropic-ai/jax_fid.git
uv sync
uv pip install jax[cuda12]  # required on the cluster for GPU support
```

The `jax_fid` repo is required for FID evaluation during training and is not available as a pip package.

## Usage

### Training

Edit hyperparameters in `main.py`, then submit to the cluster:

```bash
sbatch main.sh
```

Checkpoints and training logs are saved to `checkpoints/` by default.

### Generation

Generate digits via annealed sampling from a trained checkpoint:

```bash
sbatch generate.sh
```

Edit `generate.py` to change the checkpoint path, number of images, or annealing schedule.

## Project Structure

| File | Description |
|------|-------------|
| `config.py` | `TrainConfig` dataclass with all hyperparameters |
| `main.py` | Entry point -- instantiates config and launches training |
| `train.py` | Training loop with FID evaluation |
| `generate.py` | Annealed sampling and visualization |
| `graph.py` | Z1 graph construction and coloring |
| `model.py` | Model init, generation, and checkpointing helpers |
| `data.py` | MNIST data loading |
