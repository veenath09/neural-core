import torch

from train.action_value_heads import ActionValueHeads
from train.online_learner import OnlineLearner


def test_checkpoint_preserves_research_metadata(tmp_path):
    learner = OnlineLearner(
        gamma=0.95,
    )

    heads = ActionValueHeads(
        d=128,
        card_dim=6,
    )

    optimizer = torch.optim.Adam(
        heads.parameters(),
        lr=0.001,
    )

    metadata = {
        "component": "C2",
        "experiment": "online_learning",
        "seed": 42,
        "score_config_version": 1,
    }

    checkpoint_path = (
        tmp_path / "c2_checkpoint.pt"
    )

    learner.save_checkpoint(
        checkpoint_path,
        heads,
        optimizer,
        metadata=metadata,
    )

    restored_learner = OnlineLearner()

    restored_heads = ActionValueHeads(
        d=128,
        card_dim=6,
    )

    restored_optimizer = torch.optim.Adam(
        restored_heads.parameters(),
        lr=0.001,
    )

    restored_metadata = (
        restored_learner.load_checkpoint(
            checkpoint_path,
            restored_heads,
            restored_optimizer,
        )
    )

    assert restored_metadata == metadata