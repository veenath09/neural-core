import numpy as np
import pytest

from data.routerbench import OutcomeTable


def make_table(n=300, n_types=2, seed=0) -> OutcomeTable:
    """Synthetic outcome table: stronger (higher-tier) models succeed more often and cost more."""
    rng = np.random.default_rng(seed)
    correct = rng.random((n, 4)) < np.array([0.3, 0.5, 0.7, 0.9])
    cost = np.array([1.0, 2.0, 4.0, 8.0]) * 1e-4 * rng.uniform(0.5, 1.5, (n, 1))
    return OutcomeTable(prompt_ids=np.array([f"p{i}" for i in range(n)]),
                        step_type=np.arange(n) % n_types, correct=correct, cost=cost)


@pytest.fixture
def table():
    return make_table()
