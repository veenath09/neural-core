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

def encode_tool_branch_actions(
    strategy: Strategy,
    tool_catalog: list[str],
) -> dict[str, int]:
    encoded = encode_tool_actions(
        strategy=strategy,
        tool_catalog=tool_catalog,
    )

    return {
        f"tool::{tool_name}": selected
        for tool_name, selected in zip(
            tool_catalog,
            encoded,
        )
    }

def encode_tool_actions(
    strategy: Strategy,
    tool_catalog: list[str],
) -> list[int]:
    if len(tool_catalog) != len(set(tool_catalog)):
        raise ValueError("tool_catalog contains duplicate tools")

    unknown_tools = set(strategy.tools) - set(tool_catalog)

    if unknown_tools:
        raise ValueError(
            f"strategy contains tools not in catalog: {sorted(unknown_tools)}"
        )

    selected_tools = set(strategy.tools)

    return [
        1 if tool_name in selected_tools else 0
        for tool_name in tool_catalog
    ]


def encode_factored_strategy_actions(
    strategy: Strategy,
    tool_catalog: list[str],
) -> dict[str, int]:
    actions = encode_strategy_actions(strategy)

    actions.update(
        encode_tool_branch_actions(
            strategy=strategy,
            tool_catalog=tool_catalog,
        )
    )

    return actions