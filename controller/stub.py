"""Random but schema-valid controller, published on day 1 so other components can integrate."""

from __future__ import annotations

import random

from controller.contracts import MAX_ROUNDS, ModelCard, Strategy, TaskRep


def stub_strategy(rep: TaskRep, pool: list[ModelCard], rng: random.Random | None = None) -> Strategy:
    rng = rng or random.Random()
    halt = rng.choice(("stop", "escalate")) if rep.round < MAX_ROUNDS else "stop"
    return Strategy(step_id=rep.step_id, round=rep.round, model_id=rng.choice(pool).model_id,
                    halt=halt, complexity=rng.random(), confidence=rng.random())
