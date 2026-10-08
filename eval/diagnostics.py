"""Residual and memory diagnostics (plan §7 outputs 4 and 5).

    python -m eval.diagnostics runs/full-s0"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.episodes import Skeleton
from train.rollout import World, rollout
from train.train import load_data, load_model, make_episodes


def residual_profile(model, skeletons: Sequence[Skeleton], world: World, n: int = 256) -> list[tuple[str, float]]:
    """Mean step-to-step change of z_L / z_H over the 8 inner updates, averaged over every pass."""
    with torch.no_grad():
        out = rollout(model, make_episodes(skeletons[:n], world), world, record=True)
    kinds = [k for k, _ in out.residuals[0]]
    values = np.mean([[v for _, v in r] for r in out.residuals], axis=0)
    return list(zip(kinds, values.tolist()))


def plot_residuals(profile: list[tuple[str, float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, l_i, h_i = [], 0, 0
    for kind, _ in profile:
        if kind == "L":
            l_i += 1
            labels.append(f"L{l_i}")
        else:
            h_i += 1
            labels.append(f"H{h_i}")
    xs = np.arange(len(profile))
    ys = [v for _, v in profile]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(xs, ys, color="0.6", lw=1)
    for x, (kind, y) in zip(xs, profile):
        ax.scatter(x, y, color="#0d7a6f" if kind == "L" else "#3a55a4", marker="o" if kind == "L" else "s", zorder=3)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("mean ‖Δz‖ per update")
    ax.set_title("Inner-loop residuals (L: circles, H: squares)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def memory_trace(model, skeletons: Sequence[Skeleton], world: World, episode_id: str) -> pd.DataFrame:
    sk = next(s for s in skeletons if s.episode_id == episode_id)
    with torch.no_grad():
        out = rollout(model, make_episodes([sk], world), world, record=True)
    return pd.DataFrame(out.log)


def main(run_dir: str) -> None:
    run = Path(run_dir)
    data = load_data()
    model = load_model(run, data)
    profile = residual_profile(model, data.val, data.world)
    plot_residuals(profile, run / "residuals.png")
    (run / "residuals.json").write_text(json.dumps(profile, indent=2))
    if model.memory is not None:
        with torch.no_grad():
            out = rollout(model, make_episodes(data.val[:512], data.world), data.world, record=True)
        pd.DataFrame(out.log).to_csv(run / "memory_trace.csv", index=False)
    print(json.dumps(profile))


if __name__ == "__main__":
    main(sys.argv[1])
