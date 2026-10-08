"""RouterBench 0-shot -> one row per prompt with pool outcomes, a leakage-safe split and ModelCards
(plan §4.1, D3, D10). Pool columns are ordered cheapest first, so model index == tier."""

from __future__ import annotations

import ast
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from controller.contracts import ModelCard

HF_REPO = "withmartian/routerbench"
HF_FILE = "routerbench_0shot.pkl"
RAW_DIR = Path("data/raw")  # downloads live in the project, not the Hugging Face cache
DATA_DIR = Path("data/processed")

# Sources with 0/1 scores (D3). grade-school-math scores come in 0.25 steps and MT-Bench is graded.
SOURCES = ("mmlu", "hellaswag", "arc-challenge", "winogrande", "mbpp")
POOL = (
    "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat",
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
)
PRICE_SOURCE = "RouterBench 0-shot per-prompt total_cost"
PRICE_DATE = "not stated by RouterBench"


@dataclass
class OutcomeTable:
    """Array view of the prompt table used by the simulator."""

    prompt_ids: np.ndarray
    step_type: np.ndarray
    correct: np.ndarray  # bool [n_prompts, n_models]
    cost: np.ndarray  # float [n_prompts, n_models]
    index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self):
        self.index = {p: i for i, p in enumerate(self.prompt_ids)}

    @property
    def n_models(self) -> int:
        return self.correct.shape[1]

    @property
    def band(self) -> np.ndarray:
        return (~self.correct).sum(axis=1)

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> OutcomeTable:
        n = sum(c.startswith("correct_") for c in df.columns)
        return cls(
            prompt_ids=df["prompt_id"].to_numpy(),
            step_type=df["step_type"].to_numpy(),
            correct=df[[f"correct_{i}" for i in range(n)]].to_numpy(bool),
            cost=df[[f"cost_{i}" for i in range(n)]].to_numpy(float),
        )


def source_of(eval_name: pd.Series) -> pd.Series:
    return eval_name.where(~eval_name.str.startswith("mmlu-"), "mmlu")


def parse_prompt(raw) -> str:
    """RouterBench stores prompts as a stringified list of message parts."""
    try:
        parts = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return str(raw)
    if isinstance(parts, (list, tuple)):
        return "\n".join(str(p) for p in parts)
    return str(parts)


def build_table(raw: pd.DataFrame, pool=POOL, sources=SOURCES) -> pd.DataFrame:
    src = source_of(raw["eval_name"])
    df = raw[src.isin(sources)].assign(source=src)
    df = df.dropna(subset=list(pool) + [f"{m}|total_cost" for m in pool])
    if not np.isin(df[list(pool)].to_numpy(), (0.0, 1.0)).all():
        raise ValueError("non-binary scores in a kept source")
    out = pd.DataFrame({
        "prompt_id": df["sample_id"].astype(str).to_numpy(),
        "source": df["source"].to_numpy(),
        "step_type": df["source"].map({s: i for i, s in enumerate(sources)}).to_numpy(),
        "text": df["prompt"].map(parse_prompt).to_numpy(),
    })
    for i, m in enumerate(pool):
        out[f"correct_{i}"] = df[m].to_numpy().astype(bool)
        out[f"cost_{i}"] = df[f"{m}|total_cost"].to_numpy(float)
    out = out.drop_duplicates("text", keep="first").reset_index(drop=True)
    out["band"] = (~out[[f"correct_{i}" for i in range(len(pool))]].to_numpy()).sum(axis=1)
    return out


def split_prompts(table: pd.DataFrame, seed: int = 0, test_frac: float = 0.2,
                  val_frac: float = 0.1) -> pd.Series:
    """Split prompts once, stratified by source, before any episode is built (§4.1 step 4)."""
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=table.index, dtype=object)
    for _, idx in sorted(table.groupby("source").groups.items()):
        idx = rng.permutation(np.asarray(idx))
        n_test = round(len(idx) * test_frac)
        n_val = round((len(idx) - n_test) * val_frac)
        split.loc[idx[:n_test]] = "test"
        split.loc[idx[n_test:n_test + n_val]] = "val"
    return split


def make_model_cards(table: pd.DataFrame, pool=POOL, sources=SOURCES) -> list[ModelCard]:
    train = table[table["split"] == "train"]
    cards = []
    for i, m in enumerate(pool):
        acc = [float(train.loc[train["step_type"] == t, f"correct_{i}"].mean()) for t in range(len(sources))]
        mean_cost = float(train[f"cost_{i}"].mean())
        cards.append(ModelCard(model_id=m, provider=m.split("/")[0] if "/" in m else "openai", tier=i,
                               mean_cost_usd=mean_cost, price_source=PRICE_SOURCE, price_date=PRICE_DATE,
                               capability=acc + [math.log10(mean_cost)]))
    costs = [c.mean_cost_usd for c in cards]
    if costs != sorted(costs):
        raise ValueError(f"pool must be ordered cheapest first, got mean costs {costs}")
    return cards


def fetch_raw(raw_dir: Path = RAW_DIR) -> Path:
    """Download the RouterBench pickle into raw_dir once; reuse it afterwards."""
    raw_dir = Path(raw_dir)
    path = raw_dir / HF_FILE
    if not path.exists():
        import huggingface_hub

        raw_dir.mkdir(parents=True, exist_ok=True)
        huggingface_hub.hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset", local_dir=raw_dir)
    return path


def prepare(data_dir: Path = DATA_DIR, raw_dir: Path = RAW_DIR, seed: int = 0):
    table = build_table(pd.read_pickle(fetch_raw(raw_dir)))
    table["split"] = split_prompts(table, seed)
    cards = make_model_cards(table)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(data_dir / "prompts.parquet", index=False)
    (data_dir / "pool.json").write_text(json.dumps([asdict(c) for c in cards], indent=2))
    return table, cards


def load_prepared(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, list[ModelCard]]:
    data_dir = Path(data_dir)
    table = pd.read_parquet(data_dir / "prompts.parquet")
    cards = [ModelCard(**d) for d in json.loads((data_dir / "pool.json").read_text())]
    return table, cards


if __name__ == "__main__":
    table, cards = prepare()
    print(table.groupby(["source", "split"]).size().unstack())
    for c in cards:
        print(f"{c.tier} {c.model_id:32s} mean_cost={c.mean_cost_usd:.2e} acc={[round(a, 3) for a in c.capability[:-1]]}")
