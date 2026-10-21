import numpy as np
import torch

from controller.embedding_cache import EmbeddingCache
from controller.features import CHOOSE, HALT, build_view, collate, view
from controller.model import CONFIGS, Controller
from tests.test_env import IDS, episode

EMB = {f"p{i}": i for i in range(10)}
MIDX = {m: i for i, m in enumerate(IDS)}
ROWS = [[0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1]]


def make_model(n_cards=4, seed=0):
    torch.manual_seed(seed)
    return Controller(CONFIGS["full"], torch.randn(10, 384), torch.rand(n_cards, 6), n_types=2)


def view_kwargs(ep, s, phase, n_written):
    """Mirror what a live caller would resolve before calling build_view."""
    sk, n = ep.skeleton, ep.n_steps
    return dict(task_row=EMB[sk.prompt_ids[0]], step_row=EMB[sk.prompt_ids[s.slot]],
                step_type=int(ep.table.step_type[ep.row(s)]),
                phase=phase, rounds=s.rounds, budget_total=sk.budget_usd, budget_left=s.budget_left,
                spent=s.spent, step_index=s.step_index, n_steps=n, trace=list(s.trace),
                n_written=n_written, emb_index=EMB, model_index=MIDX)


def test_build_view_matches_view():
    for rows in (ROWS, [[1, 1, 1, 1], [0, 0, 1, 1], [1, 0, 0, 1]]):
        ep = episode("random", rows)
        states = []
        s = ep.initial_state()
        states.append(s)
        s = ep.play(s, 0)
        states.append(s)
        s = ep.end_step(s)
        states.append(s)
        s = ep.play(s, 1)
        states.append(s)
        s = ep.end_step(ep.play(s, 2))
        states.append(s)
        for s in states:
            if s.done:
                continue
            for phase in (CHOOSE, HALT):
                expected = view(ep, s, phase, 0, EMB, MIDX)
                got = build_view(**view_kwargs(ep, s, phase, 0))
                assert got == expected


def test_add_embeddings_returns_new_rows():
    m = make_model()
    rows = m.add_embeddings(torch.randn(2, 384))
    assert rows == [10, 11]
    ep = episode("random", ROWS)
    s = ep.initial_state()
    v = view(ep, s, CHOOSE, 0, EMB, MIDX)
    v = type(v)(**{**v.__dict__, "task_row": 11, "step_row": 10})
    b = collate([v])
    out = m(b, m.init_memory(1))
    assert out.model_logits.shape == (1, 4)


def test_set_cards_grows_pool():
    m = make_model(n_cards=4)
    m.set_cards(torch.rand(5, 6))
    b = collate([view(episode("random", ROWS), episode("random", ROWS).initial_state(), CHOOSE, 0, EMB, MIDX)])
    out = m(b, m.init_memory(1))
    assert out.model_logits.shape == (1, 5)


def test_set_cards_rejects_width_change():
    m = make_model(n_cards=4)
    try:
        m.set_cards(torch.rand(5, 7))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_set_cards_keeps_buffer_non_persistent_and_dtype_device():
    m = make_model(n_cards=4)
    m.set_cards(torch.rand(5, 6).double())
    assert "cards" not in m.state_dict()
    assert m.cards.dtype == torch.float32


def test_add_embeddings_keeps_buffer_non_persistent():
    m = make_model()
    m.add_embeddings(torch.randn(2, 384))
    assert "emb" not in m.state_dict()
    assert m.emb.shape == (12, 384)


def test_embedding_cache_embeds_once():
    m = make_model()
    calls = []

    def embed_fn(texts):
        calls.append(list(texts))
        return np.random.randn(len(texts), 384).astype(np.float32)

    cache = EmbeddingCache(m, dict(EMB), embed_fn)
    r1 = cache.row("x", "text")
    r2 = cache.row("x", "text")
    assert r1 == r2 == 10
    assert len(calls) == 1


def test_embedding_cache_known_key_reuses_row():
    m = make_model()
    cache = EmbeddingCache(m, dict(EMB), None)
    assert cache.row("p0", None) == 0


def test_embedding_cache_unknown_key_without_embed_fn_raises():
    m = make_model()
    cache = EmbeddingCache(m, dict(EMB), None)
    try:
        cache.row("unknown", None)
        assert False, "expected KeyError"
    except KeyError:
        pass
    try:
        cache = EmbeddingCache(m, dict(EMB), lambda texts: np.zeros((len(texts), 384), dtype=np.float32))
        cache.row("unknown", None)
        assert False, "expected KeyError"
    except KeyError:
        pass
