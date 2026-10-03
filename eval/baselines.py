"""Baseline numbers on a split: python -m eval.baselines [split] [data_dir] [out_dir]

    python -m eval.baselines test data/processed/depth_pool runs/depth
"""

from __future__ import annotations

import sys
from pathlib import Path

from data.episodes import load_skeletons
from data.routerbench import DATA_DIR, OutcomeTable, load_prepared
from eval.policies import AlwaysCheapest, AlwaysStrongest, EpisodeOracle, RandomPolicy, evaluate


def main(split: str = "test", data_dir: str = DATA_DIR, out_dir: str = "runs") -> None:
    prompts, cards = load_prepared(Path(data_dir))
    table = OutcomeTable.from_frame(prompts)
    out_dir = Path(out_dir)
    df = evaluate(load_skeletons(split, Path(data_dir)), table, [c.model_id for c in cards],
                  [AlwaysCheapest(), AlwaysStrongest(), RandomPolicy(0), EpisodeOracle()])
    wide = df.pivot(index="episode_id", columns="policy", values="quality")
    for name in ("always_cheapest", "always_strongest", "random"):
        assert (wide["oracle"] >= wide[name]).all(), f"oracle below {name} on some episode"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"baselines_{split}.csv", index=False)
    print(df.groupby("policy")[["quality", "cost_usd"]].mean().to_string(float_format="%.4f"))
    print(df.pivot_table(index="ep_type", columns="policy", values="quality").to_string(float_format="%.3f"))


if __name__ == "__main__":
    main(*sys.argv[1:4])
