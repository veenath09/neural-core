"""Embed every prompt once with frozen bge-small and cache to .npy (plan §4.1 step 5)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from data.routerbench import CACHE_DIR, load_prepared

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384

Encoder = Callable[[list[str]], np.ndarray]


def sentence_transformer_encoder(name: str = MODEL_NAME, device: str | None = None) -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name, device=device)
    return lambda texts: model.encode(texts, batch_size=64, normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False)


def build_cache(ids: Sequence[str], texts: Sequence[str], encode: Encoder, cache_dir: Path = CACHE_DIR,
                batch_size: int = 1024, dim: int = DIM) -> np.ndarray:
    chunks = []
    for start in range(0, len(texts), batch_size):
        chunks.append(np.asarray(encode(list(texts[start:start + batch_size])), dtype=np.float32))
        print(f"embedded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    emb = np.concatenate(chunks)
    if emb.shape != (len(ids), dim):
        raise ValueError(f"embedding shape {emb.shape} != {(len(ids), dim)}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "embeddings.npy", emb)
    (cache_dir / "embedding_ids.json").write_text(json.dumps(list(ids)))
    return emb


def load_cache(cache_dir: Path = CACHE_DIR) -> tuple[np.ndarray, dict[str, int]]:
    cache_dir = Path(cache_dir)
    emb = np.load(cache_dir / "embeddings.npy", mmap_mode="r")
    ids = json.loads((cache_dir / "embedding_ids.json").read_text())
    return emb, {p: i for i, p in enumerate(ids)}


if __name__ == "__main__":
    table, _ = load_prepared()
    build_cache(table["prompt_id"].tolist(), table["text"].tolist(), sentence_transformer_encoder())
    start = time.perf_counter()
    emb, index = load_cache()
    print(f"loaded {emb.shape} in {time.perf_counter() - start:.3f}s")
