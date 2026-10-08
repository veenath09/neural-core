"""Bucketed return baseline for Monte Carlo RL (plan C2-05).

Table b[band, lam_idx], zeros at the start, updated by an exponential moving
average (alpha=0.05) toward the mean return observed in each (band, lam_idx)
bucket in a batch. Fixed shape [n_bands, n_lams] so it survives a growing
model pool (band is a fixed 0..4 bucket, not a model count). Saved with the
checkpoint via state_dict/load_state_dict because training continues it.
"""

from __future__ import annotations

import torch


class BucketedBaseline:
    def __init__(self, n_bands: int = 5, n_lams: int = 5, alpha: float = 0.05):
        self.n_bands = n_bands
        self.n_lams = n_lams
        self.alpha = alpha
        self.table = torch.zeros(n_bands, n_lams)

    def value(self, band: torch.Tensor, lam_idx: torch.Tensor) -> torch.Tensor:
        """Baseline value per slot: table[band, lam_idx], shape like band/lam_idx.

        band/lam_idx may live on any device (e.g. the trace's accelerator
        device); they are moved to self.table's device internally, and the
        looked-up values come back on that same device (the table's), so the
        caller is responsible for moving the result to wherever it combines it.
        """
        device = self.table.device
        return self.table[band.to(device), lam_idx.to(device)]

    def update(self, band: torch.Tensor, lam_idx: torch.Tensor, returns: torch.Tensor) -> None:
        """EMA each bucket present toward the mean return observed for it.

        b <- b + alpha * (mean(returns in bucket) - b), only for buckets that
        appear at least once; other buckets are left untouched. band/lam_idx/
        returns may live on any device; they are moved to self.table's device
        internally.
        """
        device = self.table.device
        band = band.reshape(-1).long().to(device)
        lam_idx = lam_idx.reshape(-1).long().to(device)
        returns = returns.reshape(-1).float().to(device)

        flat_idx = band * self.n_lams + lam_idx
        n_buckets = self.n_bands * self.n_lams

        sums = torch.zeros(n_buckets, dtype=returns.dtype, device=device)
        counts = torch.zeros(n_buckets, dtype=returns.dtype, device=device)
        sums.scatter_add_(0, flat_idx, returns)
        counts.scatter_add_(0, flat_idx, torch.ones_like(returns))

        present = counts > 0
        means = torch.zeros(n_buckets, dtype=returns.dtype, device=device)
        means[present] = sums[present] / counts[present]

        flat_table = self.table.reshape(-1)
        flat_table[present] = flat_table[present] + self.alpha * (means[present] - flat_table[present])
        self.table = flat_table.reshape(self.n_bands, self.n_lams)

    def state_dict(self) -> dict:
        return {
            "table": self.table.clone(),
            "n_bands": self.n_bands,
            "n_lams": self.n_lams,
            "alpha": self.alpha,
        }

    def load_state_dict(self, d: dict) -> None:
        self.table = d["table"].clone()
        self.n_bands = d["n_bands"]
        self.n_lams = d["n_lams"]
        self.alpha = d["alpha"]
