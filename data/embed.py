"""Embed every prompt once with frozen bge-small and cache to .npy (plan §4.1 step 5)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from data.routerbench import DATA_DIR, RAW_DIR, load_prepared

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384
MODEL_DIR = RAW_DIR / "bge-small-en-v1.5"

Encoder = Callable[[list[str]], np.ndarray]


def fetch_encoder(model_dir: Path = MODEL_DIR) -> Path:
    """Download bge-small into model_dir once; reuse it afterwards."""
    model_dir = Path(model_dir)
    if not (model_dir / "config.json").exists():
        import huggingface_hub

        model_dir.mkdir(parents=True, exist_ok=True)
        huggingface_hub.snapshot_download(MODEL_NAME, local_dir=model_dir)
    return model_dir


def sentence_transformer_encoder(model_dir: Path = MODEL_DIR, device: str | None = None) -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(fetch_encoder(model_dir)), device=device)
    return lambda texts: model.encode(texts, batch_size=64, normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False)


def build_cache(ids: Sequence[str], texts: Sequence[str], encode: Encoder, data_dir: Path = DATA_DIR,
                batch_size: int = 1024, dim: int = DIM) -> np.ndarray:
    chunks = []
    for start in range(0, len(texts), batch_size):
        chunks.append(np.asarray(encode(list(texts[start:start + batch_size])), dtype=np.float32))
        print(f"embedded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    emb = np.concatenate(chunks)
    if emb.shape != (len(ids), dim):
        raise ValueError(f"embedding shape {emb.shape} != {(len(ids), dim)}")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "embeddings.npy", emb)
    (data_dir / "embedding_ids.json").write_text(json.dumps(list(ids)))
    return emb


def load_cache(data_dir: Path = DATA_DIR) -> tuple[np.ndarray, dict[str, int]]:
    data_dir = Path(data_dir)
    emb = np.load(data_dir / "embeddings.npy", mmap_mode="r")
    ids = json.loads((data_dir / "embedding_ids.json").read_text())
    return emb, {p: i for i, p in enumerate(ids)}


if __name__ == "__main__":
    table, _ = load_prepared()
    build_cache(table["prompt_id"].tolist(), table["text"].tolist(), sentence_transformer_encoder())
    start = time.perf_counter()
    emb, index = load_cache()
    print(f"loaded {emb.shape} in {time.perf_counter() - start:.3f}s")
