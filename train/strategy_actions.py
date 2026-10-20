from __future__ import annotations

from controller.contracts import Strategy


MEMORY_TO_INDEX = {
    "none": 0,
    "session": 1,
    "retrieval": 2,
}

DEPTH_TO_INDEX = {
    "low": 0,
    "medium": 1,
    "high": 2,
}

VERIFICATION_TO_INDEX = {
    "none": 0,
    "self_check": 1,
    "cross_model": 2,
}

TOPOLOGY_TO_INDEX = {
    "single": 0,
    "parallel": 1,
    "sub_agents": 2,
}


def encode_strategy_actions(
    strategy: Strategy,
) -> dict[str, int]:
    return {
        "memory": MEMORY_TO_INDEX[strategy.memory],
        "depth": DEPTH_TO_INDEX[strategy.depth],
        "verification": VERIFICATION_TO_INDEX[
            strategy.verification
        ],
        "topology": TOPOLOGY_TO_INDEX[
            strategy.topology
        ],
    }