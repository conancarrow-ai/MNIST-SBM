"""Plot training samples saved during checkpointing."""

import glob
import os

import numpy as np
import matplotlib.pyplot as plt

files = sorted(glob.glob("checkpoints/samples_epoch_*.npy"))
if not files:
    print("No sample files found")
else:
    for f in files:
        samples = np.load(f)
        n = samples.shape[0]
        fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
        if n == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.imshow(samples[i].reshape(28, 28), cmap="gray")
            ax.axis("off")
        fig.suptitle(os.path.basename(f))
        plt.tight_layout()
        epoch = os.path.splitext(os.path.basename(f))[0]
        plt.savefig(f"Trained_SBM_Samples_{epoch}.png", dpi=150, bbox_inches="tight")
        print(f"Saved Trained_SBM_Samples_{epoch}.png", flush=True)
        plt.close(fig)
