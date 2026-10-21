import numpy as np
import pytest

from eval.final_test import t_interval


def test_t_interval_matches_textbook():
    out = t_interval(np.array([1.0, 2.0, 3.0]))
    assert out["mean"] == 2.0 and out["n"] == 3
    half = 4.302652729911275 * 1.0 / np.sqrt(3)  # t(0.975, 2) * sd / sqrt(n)
    assert out["ci95"] == pytest.approx([2 - half, 2 + half])


def test_single_seed_has_no_interval():
    assert t_interval(np.array([0.5]))["ci95"] is None
