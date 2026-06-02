from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_loss(history1, history2=None, out_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history1["train_nll"], label="task1 train")
    ax.plot(history1["valid_nll"], label="task1 valid")
    if history2:
        ax.plot(history2["train_nll"], label="task2 train")
        ax.plot(history2["valid_nll"], label="task2 valid")
    ax.set_xlabel("epoch")
    ax.set_ylabel("NLL")
    ax.legend()
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
    return fig


def plot_piano_roll(grid, title=None, out_path=None):
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for v in range(min(4, grid.shape[1])):
        y = np.where(grid[:, v] >= 0, grid[:, v], np.nan)
        ax.plot(y, ".", ms=3, color=colors[v], label=["S", "A", "T", "B"][v])
    ax.set_xlabel("16th-note frame")
    ax.set_ylabel("MIDI pitch")
    if title:
        ax.set_title(title)
    ax.legend(ncol=4, loc="upper right")
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
    return fig


def overlay_hist(series, labels, title, out_path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    for arr, label in zip(series, labels):
        ax.plot(arr, marker="o", label=label)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
    return fig

