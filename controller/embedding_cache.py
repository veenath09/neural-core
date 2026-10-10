"""Run-time embedding of unseen live prompts: embed once, reuse the row (plan Part 2 Task 1)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from controller.model import Controller


class EmbeddingCache:
    """Wraps a Controller's embedding table with a key -> row index cache.

    A known key reuses its existing row. An unknown key is embedded once with
    `embed_fn` (when given, together with its text), appended to the model's
    embedding buffer, and its row recorded for next time."""

    def __init__(self, model: Controller, emb_index: dict[str, int],
                embed_fn: Callable[[list[str]], np.ndarray] | None):
        self.model = model
        self.emb_index = emb_index
        self.embed_fn = embed_fn

    def row(self, key: str, text: str | None) -> int:
        if key in self.emb_index:
            return self.emb_index[key]
        if text is None or self.embed_fn is None:
            raise KeyError(key)
        vec = np.asarray(self.embed_fn([text]), dtype=np.float32)
        rows = self.model.add_embeddings(torch.from_numpy(vec))
        row = rows[0]
        self.emb_index[key] = row
        return row
