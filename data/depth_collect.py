"""
Purpose:  Collect per-prompt outcomes for LLMs at several reasoning depths, so the controller can
          learn when thinking harder pays off (RouterBench records one answer per model only).
Owner:    C1 Controller (V M Wickramaarachchi)
Contract: v1
Inputs:   data/processed/prompts.parquet (prompts, split), data/raw/routerbench_0shot.pkl (to recover
          gold answers), API keys from .env (never logged).
Outputs:  data/raw/depth/outcomes.jsonl, one line per (prompt_id, arm) call: correctness, tokens,
          latency, list-price cost.
Notes:    Gold answer = the letter given by RouterBench models that were scored correct (they never
          disagree on the multiple-choice sources used here). Cost is tokens x the provider's published
          list price recorded in ARMS (with the date), so free-tier calls are still priced for the
          thesis' cost axis; the spending cap is enforced on that list-price total. Resumable: calls
          already in outcomes.jsonl are skipped.

    python -m data.depth_collect --prompts 20 --cap 2      # pilot
    python -m data.depth_collect --prompts 500 --cap 20    # full run (stops at the cap with whole prompts)
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from data.routerbench import DATA_DIR, RAW_DIR

MC_SOURCES = ("mmlu", "hellaswag", "arc-challenge", "winogrande")
OUT_PATH = RAW_DIR / "depth" / "outcomes.jsonl"
PRICE_DATE = "2026-10-19"
MAX_OUTPUT_TOKENS = 8192


@dataclass(frozen=True)
class Arm:
    """One (model, depth) option. Prices are list prices in USD per 1M tokens, output incl. thinking."""

    name: str
    provider: str  # groq | gemini | openrouter | kimi | openai | anthropic
    model: str
    depth: str  # off | minimal | low | medium | high
    price_in: float
    price_out: float
    params: dict = field(default_factory=dict)


ARMS: tuple[Arm, ...] = (
    # cheapest first; prices are list prices (USD / 1M tokens) on PRICE_DATE
    Arm("gpt-oss-20b@low", "groq", "openai/gpt-oss-20b", "low", 0.018, 0.09, {"reasoning_effort": "low"}),
    Arm("gpt-oss-20b@medium", "groq", "openai/gpt-oss-20b", "medium", 0.018, 0.09, {"reasoning_effort": "medium"}),
    Arm("gpt-oss-20b@high", "groq", "openai/gpt-oss-20b", "high", 0.018, 0.09, {"reasoning_effort": "high"}),
    Arm("deepseek-v4.1-flash@off", "openrouter", "deepseek/deepseek-v4.1-flash", "off", 0.033, 0.08,
        {"reasoning": {"enabled": False}}),
    Arm("deepseek-v4.1-flash@high", "openrouter", "deepseek/deepseek-v4.1-flash", "high", 0.033, 0.08,
        {"reasoning": {"effort": "high"}}),
    Arm("gpt-6-luna@off", "openai", "gpt-6-luna", "off", 0.10, 0.50, {"reasoning_effort": "none"}),
    Arm("gpt-6-luna@low", "openai", "gpt-6-luna", "low", 0.10, 0.50, {"reasoning_effort": "low"}),
    Arm("gpt-6-luna@medium", "openai", "gpt-6-luna", "medium", 0.10, 0.50, {"reasoning_effort": "medium"}),
    Arm("gpt-6-luna@high", "openai", "gpt-6-luna", "high", 0.10, 0.50, {"reasoning_effort": "high"}),
    Arm("gemini-3.5-flash-lite@minimal", "gemini", "gemini-3.5-flash-lite", "minimal", 0.30, 2.50,
        {"thinkingLevel": "minimal"}),
    Arm("gemini-3.5-flash-lite@medium", "gemini", "gemini-3.5-flash-lite", "medium", 0.30, 2.50,
        {"thinkingLevel": "medium"}),
    Arm("gemini-3.5-flash-lite@high", "gemini", "gemini-3.5-flash-lite", "high", 0.30, 2.50, {"thinkingLevel": "high"}),
    Arm("gpt-oss-120b@low", "groq", "openai/gpt-oss-120b", "low", 0.15, 0.60, {"reasoning_effort": "low"}),
    Arm("gpt-oss-120b@high", "groq", "openai/gpt-oss-120b", "high", 0.15, 0.60, {"reasoning_effort": "high"}),
    Arm("gemini-3.1-flash-lite@minimal", "gemini", "gemini-3.1-flash-lite", "minimal", 0.25, 1.50,
        {"thinkingLevel": "minimal"}),
    Arm("gemini-3.1-flash-lite@medium", "gemini", "gemini-3.1-flash-lite", "medium", 0.25, 1.50,
        {"thinkingLevel": "medium"}),
    Arm("gemini-3.1-flash-lite@high", "gemini", "gemini-3.1-flash-lite", "high", 0.25, 1.50, {"thinkingLevel": "high"}),
    Arm("gemini-3.8-flash@low", "gemini", "gemini-3.8-flash", "low", 0.75, 3.75, {"thinkingLevel": "low"}),
    Arm("gemini-3.8-flash@medium", "gemini", "gemini-3.8-flash", "medium", 0.75, 3.75, {"thinkingLevel": "medium"}),
    Arm("gemini-3.8-flash@high", "gemini", "gemini-3.8-flash", "high", 0.75, 3.75, {"thinkingLevel": "high"}),
    Arm("kimi-k2.6@on", "kimi", "kimi-k2.6", "high", 0.65, 3.41, {}),
    Arm("claude-sonnet-5@off", "anthropic", "claude-sonnet-5", "off", 2.00, 10.00, {"thinking": {"type": "disabled"}}),
    Arm("claude-sonnet-5@low", "anthropic", "claude-sonnet-5", "low", 2.00, 10.00,
        {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}),
    Arm("claude-sonnet-5@medium", "anthropic", "claude-sonnet-5", "medium", 2.00, 10.00,
        {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}),
    Arm("claude-sonnet-5@high", "anthropic", "claude-sonnet-5", "high", 2.00, 10.00,
        {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}),
    Arm("gemini-3.1-pro@low", "gemini", "gemini-3.1-pro-preview", "low", 2.00, 12.00, {"thinkingLevel": "low"}),
    Arm("gemini-3.1-pro@medium", "gemini", "gemini-3.1-pro-preview", "medium", 2.00, 12.00,
        {"thinkingLevel": "medium"}),
    Arm("gemini-3.1-pro@high", "gemini", "gemini-3.1-pro-preview", "high", 2.00, 12.00, {"thinkingLevel": "high"}),
)


@dataclass
class Outcome:
    prompt_id: str
    arm: str
    provider: str
    model: str
    depth: str
    split: str
    source: str
    gold: str
    answer: str | None
    correct: bool
    input_tokens: int
    output_tokens: int  # visible answer tokens
    reasoning_tokens: int
    latency_ms: float
    cost_usd: float  # list price x tokens
    price_date: str
    error: str | None = None
    ts: float = 0.0


# ---------- pure helpers (tested) ----------

_LETTER = re.compile(r"\b([A-D])\b")


def parse_answer(text: str | None, choices: str = "ABCD") -> str | None:
    """Final answer letter: an explicit 'Answer: X' wins, otherwise the last standalone letter."""
    if not text:
        return None
    explicit = re.findall(r"[Aa]nswer\s*[:\-]?\s*\(?\**\s*([A-D])\b", text)
    letters = explicit or _LETTER.findall(text)
    letters = [x for x in letters if x in choices]
    return letters[-1] if letters else None


def recover_gold(responses: Sequence[str | None], correct: Sequence[bool]) -> str | None:
    """Gold letter from the models RouterBench scored correct; None if none parse or they disagree."""
    letters = {parse_answer(r) for r, ok in zip(responses, correct) if ok}
    letters.discard(None)
    return letters.pop() if len(letters) == 1 else None


def call_cost(arm: Arm, input_tokens: int, output_tokens: int, reasoning_tokens: int) -> float:
    return (input_tokens * arm.price_in + (output_tokens + reasoning_tokens) * arm.price_out) / 1e6


def choices_for(source: str) -> str:
    return "AB" if source == "winogrande" else "ABCD"


def done_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("error") is None:
            keys.add((row["prompt_id"], row["arm"]))
    return keys


def spent(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(json.loads(line).get("cost_usd", 0.0) for line in path.read_text().splitlines())


class Budget:
    """Thread-safe running total; refuses new calls once the cap is reached."""

    def __init__(self, cap: float, already: float = 0.0):
        self.cap, self.total = cap, already
        self._lock = threading.Lock()

    def can_spend(self) -> bool:
        with self._lock:
            return self.total < self.cap

    def add(self, usd: float) -> None:
        with self._lock:
            self.total += usd


# ---------- prompt selection ----------

def select_prompts(n: int, seed: int = 0, data_dir: Path = DATA_DIR, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """n multiple-choice prompts with a recoverable gold letter, stratified by split and source."""
    prompts = pd.read_parquet(Path(data_dir) / "prompts.parquet")
    prompts = prompts[prompts["source"].isin(MC_SOURCES)].copy()
    raw = pd.read_pickle(Path(raw_dir) / "routerbench_0shot.pkl").set_index("sample_id")
    models = [c for c in raw.columns if "|" not in c and c not in ("prompt", "eval_name", "oracle_model_to_route_to")]
    sub = raw.loc[prompts["prompt_id"]]
    prompts["gold"] = [recover_gold([r[f"{m}|model_response"] for m in models], [r[m] == 1.0 for m in models])
                       for _, r in sub.iterrows()]
    prompts = prompts[prompts["gold"].notna()]
    shares = prompts.groupby(["split", "source"]).size() / len(prompts)
    parts = []
    for (split, source), share in shares.items():
        group = prompts[(prompts["split"] == split) & (prompts["source"] == source)]
        k = min(len(group), max(1, round(n * share)))
        parts.append(group.sample(k, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).head(n)[["prompt_id", "split", "source", "text", "gold"]]


# ---------- provider calls (thin) ----------

def load_env(path: Path = Path(".env")) -> dict[str, str]:
    env = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip("\"'")
    return env


def _post(url: str, body: dict, headers: dict, timeout: float = 300) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "User-Agent": "neural-core/0.1"} | headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _openai_style(url: str, key_headers: dict, arm: Arm, text: str) -> tuple[str | None, int, int, int]:
    body = {"model": arm.model, "messages": [{"role": "user", "content": text}],
            "max_tokens": MAX_OUTPUT_TOKENS} | arm.params
    r = _post(url, body, key_headers)
    msg = r["choices"][0]["message"]
    u = r.get("usage", {})
    reasoning = (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    completion = u.get("completion_tokens", 0)
    return msg.get("content"), u.get("prompt_tokens", 0), max(0, completion - reasoning), reasoning


def call_arm(arm: Arm, text: str, env: dict[str, str]) -> tuple[str | None, int, int, int]:
    """Returns (answer text, input tokens, visible output tokens, reasoning tokens)."""
    if arm.provider == "groq":
        return _openai_style("https://api.groq.com/openai/v1/chat/completions",
                             {"Authorization": "Bearer " + env["GROQ_API_KEY"]}, arm, text)
    if arm.provider == "openrouter":
        return _openai_style("https://openrouter.ai/api/v1/chat/completions",
                             {"Authorization": "Bearer " + env["OPEN_ROUTER_API_KEY"]}, arm, text)
    if arm.provider == "openai":
        body = {"model": arm.model, "messages": [{"role": "user", "content": text}],
                "max_completion_tokens": MAX_OUTPUT_TOKENS} | arm.params
        r = _post("https://api.openai.com/v1/chat/completions", body,
                  {"Authorization": "Bearer " + env["OPENAI_API_KEY"]})
        u = r.get("usage", {})
        reasoning = (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        return (r["choices"][0]["message"].get("content"), u.get("prompt_tokens", 0),
                max(0, u.get("completion_tokens", 0) - reasoning), reasoning)
    if arm.provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=env["ANTHROPIC_API_KEY"], max_retries=0)
        msg = client.messages.create(model=arm.model, max_tokens=MAX_OUTPUT_TOKENS,
                                     messages=[{"role": "user", "content": text}], **arm.params)
        answer = "".join(b.text for b in msg.content if b.type == "text")
        # output_tokens includes thinking; Anthropic does not split it out, so it is counted as output
        return answer, msg.usage.input_tokens, msg.usage.output_tokens, 0
    if arm.provider == "kimi":
        return _openai_style(env["KIMI_BASE_URL"].rstrip("/") + "/chat/completions",
                             {"api-key": env["KIMI_API_KEY"]}, arm, text)
    if arm.provider == "gemini":
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/{arm.model}:generateContent"
               f"?key={env['GOOGLE_API_KEY']}")
        body = {"contents": [{"role": "user", "parts": [{"text": text}]}],
                "generationConfig": {"maxOutputTokens": MAX_OUTPUT_TOKENS, "thinkingConfig": dict(arm.params)}}
        r = _post(url, body, {})
        parts = r["candidates"][0].get("content", {}).get("parts", [])
        answer = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        u = r.get("usageMetadata", {})
        return answer, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0), u.get("thoughtsTokenCount", 0)
    raise ValueError(f"unknown provider {arm.provider!r}")


# ---------- runner ----------

def run_one(arm: Arm, row: pd.Series, env: dict, call: Callable = call_arm, retries: int = 4) -> Outcome:
    err, answer_text, tin, tout, treason = None, None, 0, 0, 0
    t0 = time.perf_counter()
    for attempt in range(retries):
        try:
            answer_text, tin, tout, treason = call(arm, row["text"], env)
            err = None
            break
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            break
        except Exception as e:  # SDK errors, network errors, malformed responses
            err = f"{type(e).__name__}: {str(e)[:200]}"
            status = getattr(e, "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                break  # client error (bad request, auth, usage limit): retrying will not help
            time.sleep(min(60, 5 * 2 ** attempt))
    latency = (time.perf_counter() - t0) * 1000
    answer = parse_answer(answer_text, choices_for(row["source"]))
    return Outcome(prompt_id=row["prompt_id"], arm=arm.name, provider=arm.provider, model=arm.model, depth=arm.depth,
                   split=row["split"], source=row["source"], gold=row["gold"], answer=answer,
                   correct=answer == row["gold"], input_tokens=tin, output_tokens=tout, reasoning_tokens=treason,
                   latency_ms=latency, cost_usd=call_cost(arm, tin, tout, treason) if err is None else 0.0,
                   price_date=PRICE_DATE, error=err, ts=time.time())


def collect(prompts: pd.DataFrame, arms: Sequence[Arm], cap: float, out_path: Path = OUT_PATH,
            env: dict | None = None, call: Callable = call_arm, workers_per_provider: int = 4,
            chunk: int = 10) -> Budget:
    env = env if env is not None else load_env()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(out_path)
    budget = Budget(cap, spent(out_path))
    lock = threading.Lock()

    def work(arm: Arm, row: pd.Series) -> None:
        if not budget.can_spend():
            return
        o = run_one(arm, row, env, call)
        budget.add(o.cost_usd)
        with lock, out_path.open("a") as f:
            f.write(json.dumps(asdict(o)) + "\n")

    for start in range(0, len(prompts), chunk):  # prompt-major: a chunk gets every arm before the next chunk
        block = prompts.iloc[start:start + chunk]
        for arm in arms:  # cheapest first within a chunk
            if not budget.can_spend():
                print(f"cap ${cap:.2f} reached at prompt {start}; stopping", flush=True)
                return budget
            todo = [row for _, row in block.iterrows() if (row["prompt_id"], arm.name) not in done]
            with ThreadPoolExecutor(max_workers=workers_per_provider) as pool:
                list(pool.map(lambda r: work(arm, r), todo))
        print(f"prompts {start}-{start + len(block) - 1} done, total ${budget.total:.3f}", flush=True)
    return budget


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=20)
    ap.add_argument("--cap", type=float, default=2.0, help="USD at list price, all providers together")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", nargs="*", help="subset of arm names")
    ap.add_argument("--skip-provider", nargs="*", default=[], help="providers to leave out, e.g. anthropic")
    a = ap.parse_args()
    arms = [x for x in ARMS if (not a.arms or x.name in a.arms) and x.provider not in a.skip_provider]
    if not arms:
        raise SystemExit("no arms selected: check --arms / --skip-provider")
    prompts = select_prompts(a.prompts, a.seed)
    print(f"{len(prompts)} prompts x {len(arms)} arms, cap ${a.cap:.2f}", flush=True)
    collect(prompts, arms, a.cap)


if __name__ == "__main__":
    main()
