"""Baseline numbers on a split: python -m eval.baselines [split]"""

from __future__ import annotations

import sys
from pathlib import Path

from data.episodes import load_skeletons
from data.routerbench import OutcomeTable, load_prepared
from eval.policies import AlwaysCheapest, AlwaysStrongest, EpisodeOracle, RandomPolicy, evaluate


def main(split: str = "test", out_dir: Path = Path("runs")) -> None:
    prompts, cards = load_prepared()
    table = OutcomeTable.from_frame(prompts)
    df = evaluate(load_skeletons(split), table, [c.model_id for c in cards],
                  [AlwaysCheapest(), AlwaysStrongest(), RandomPolicy(0), EpisodeOracle()])
    wide = df.pivot(index="episode_id", columns="policy", values="quality")
    for name in ("always_cheapest", "always_strongest", "random"):
        assert (wide["oracle"] >= wide[name]).all(), f"oracle below {name} on some episode"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / f"baselines_{split}.csv", index=False)
    print(df.groupby("policy")[["quality", "cost_usd"]].mean().to_string(float_format="%.4f"))
    print(df.pivot_table(index="ep_type", columns="policy", values="quality").to_string(float_format="%.3f"))


if __name__ == "__main__":
    main(*sys.argv[1:2])
