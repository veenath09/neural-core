import numpy as np

from data.depth_synth import agreement, find_donors


def test_donor_is_same_task_with_closest_score_pattern():
    src = np.array(["mmlu", "mmlu", "gsm8k"])
    scores = np.array([[1, 1, 0], [0, 0, 0], [1, 1, 0]], dtype=float)
    emb = np.eye(3, dtype=np.float32)
    q = find_donors(np.array(["mmlu"]), np.array([[1, 1, 1]], float), np.ones((1, 3), np.float32) / 3 ** 0.5,
                    src, scores, emb)
    assert q.tolist() == [0]  # closest pattern among mmlu, never the gsm8k row with the same pattern


def test_leave_one_out_never_picks_itself():
    src = np.array(["a", "a", "a"])
    scores = np.zeros((3, 2))
    emb = np.eye(3, dtype=np.float32)
    d = find_donors(src, scores, emb, src, scores, emb, exclude_self=True)
    assert all(d != np.arange(3))


def test_agreement_lift():
    real = np.array([[True], [True], [False], [False]])
    a = agreement(real, real.copy())
    assert a["p_real_right_if_imputed_right"] == 1.0 and a["p_real_right_if_imputed_wrong"] == 0.0
