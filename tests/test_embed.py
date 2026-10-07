import time

import numpy as np
import pytest

from data.embed import DIM, build_cache, load_cache


def fake_encode(texts):
    return np.stack([np.full(DIM, len(t), dtype=np.float32) for t in texts])


def test_cache_round_trip_keeps_id_order(tmp_path):
    ids, texts = ["a", "b", "c"], ["x", "yy", "zzz"]
    build_cache(ids, texts, fake_encode, data_dir=tmp_path, batch_size=2)
    emb, index = load_cache(tmp_path)
    assert emb.shape == (3, DIM) and emb.dtype == np.float32
    assert emb[index["c"], 0] == 3.0


def test_build_cache_rejects_wrong_dim(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        build_cache(["a"], ["x"], lambda t: np.zeros((len(t), 7)), data_dir=tmp_path)


def test_load_is_fast(tmp_path):
    n = 30_000
    build_cache([str(i) for i in range(n)], ["t"] * n,
                lambda t: np.zeros((len(t), DIM), np.float32), data_dir=tmp_path, batch_size=10_000)
    start = time.perf_counter()
    emb, index = load_cache(tmp_path)
    assert time.perf_counter() - start < 1.0
    assert emb.shape == (n, DIM) and len(index) == n


def test_fetch_encoder_downloads_into_project_dir_once(tmp_path, monkeypatch):
    import huggingface_hub

    from data.embed import MODEL_NAME, fetch_encoder

    calls = []

    def fake_snapshot(repo_id, local_dir):
        calls.append(repo_id)
        (local_dir / "config.json").write_text("{}")
        return str(local_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    model_dir = tmp_path / "bge"
    assert fetch_encoder(model_dir) == model_dir
    assert fetch_encoder(model_dir) == model_dir
    assert calls == [MODEL_NAME]
