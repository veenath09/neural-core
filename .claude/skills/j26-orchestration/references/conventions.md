# Conventions

Read before your first commit. These rules exist so four people's code merges without surprises.

## Repository layout

```
common/                 shared: contracts.py, protocols.py, ids.py, io.py
controller/             C1
  embed.py              TextEmbedder (frozen bge-small, cached)
  encoder.py            NumericFeaturizer, StepEncoder
  blocks.py             MLPBlock, RMSNorm
  memory.py             TTTLinearMemory, CfCGate, SigmoidGate, ConstantGate
  core.py               HModule, LModule, HRMCore (schedule, truncated backprop)
  heads.py              ModelScoringHead, HaltingHead, ComplexityHead (+ Phase 2 heads)
  calibrate.py          temperature fitting
  model.py              Controller (decide, observe, reset)
  stub.py               random valid Strategy (Phase 0)
learning/               C2
  rollout.py            RolloutWorker, behaviour policy mixing
  supervised.py         SupervisedTrainer
  rl.py                 MonteCarloPG
  baseline.py           BucketedBaseline, ValueHead
  replay.py             ReplayBuffer (Phase 3)
  exploration.py        entropy schedule, new-model bonus
  safety.py             SafetyGuard (validation gate, rollback)
  monitor.py            alarms over logged metrics
  checkpoints.py        CheckpointManager
measurement/            C3
  routerbench.py        load, filter binary sources, choose pool, split prompts
  episodes.py           EpisodeSkeleton generator
  simulator.py          Simulator (EnvironmentAPI), outcome rules, noisy verifier
  oracle.py             per-step oracle, exact episode oracle
  scorer.py             CompositeScorer
  verifier.py           live verifiers (Phase 3)
  profiler.py           ProbeProfiler, ProfileUpdater
  baselines.py          always-strong, always-cheap, random, RouteLLM, LLM-as-router
  evaluate.py           frontier sweep, tables, plots
  stats.py              bootstrap, seed CI, Holm, frontier area (shared with the dashboard)
execution/              C4
  catalogue.py          in-memory Catalogue (Phases 0-2), DB-backed (Phase 3)
  session.py            DictSessionStore, RedisSessionStore
  records.py            ParquetRecordStore, PostgresRecordStore
  executor.py           StrategyExecutor
  providers.py          ProviderAdapter (LiteLLM)
  gateway.py            FastAPI app (Phase 3)
  adapters/             LangGraph, CrewAI, Agents SDK episode-header hooks (Phase 3)
ui/                     data.py (DuckDB views), dashboard.py (Streamlit), copied from the skill's assets
runs/                   RunLogger output (not committed)
storage/migrations/     SQL (Phase 3)
conf/                   YAML configs (copied from the skill's assets)
scripts/                entry points: prepare_data.py, train.py, evaluate.py, make_figures.py
tests/                  contracts/, controller/, learning/, measurement/, execution/, integration/
docs/                   design documents, experiment_log.md, decision_log.md
```

## Git

- `main` is always green (tests pass). Work on branches: `c1/<task>`, `c2/<task>`, `c3/<task>`, `c4/<task>`, `contract/<change>`.
- One task from your component guide = one pull request = one or a few commits.
- Commit messages: `<type>(<scope>): <summary>`, types `feat`, `fix`, `test`, `refactor`, `docs`, `chore`, `contract`. Scope is the package: `controller`, `learning`, `measurement`, `execution`, `common`.
- Pull requests touching `common/` or `conf/controller.yaml` need approval from all members.
- Never commit data, caches, checkpoints or results. Commit the scripts that regenerate them.

## Testing

- `pytest -q` must pass before every push.
- Every task in a component guide names its test. Write the test first where practical.
- Shape tests use small sizes (B=3) and assert exact shapes.
- Tests never download data or call APIs. Use fixtures in `tests/fixtures/` (a 200-prompt RouterBench sample, committed as small Parquet, is allowed).

## Reproducibility

- Every script takes `--seed` and sets `random`, `numpy` and `torch` seeds.
- Every run writes its resolved config next to its outputs.
- Every figure and table is produced by `scripts/make_figures.py` from saved predictions in `results/`, never by hand.
- Costs are reported in USD with the price date from `ModelCard.price_date`.

## Naming

- ids: `episode_id` = `{split}-{type}-{index:06d}`; `step_id` = `{episode_id}/{step_index}`; rounds are a separate integer, never baked into `step_id`.
- Config names: `full`, `gate_off`, `gate_sigmoid`, `memory_off`, `context_free`. Use these exact strings in files, tables and plots.
- Tensor shape symbols (in docstrings): B batch, d width, e text dim, k trace length, P pool size, U tools offered, F capability size, S step types.

## Style

- Python 3.12, type hints on all public functions, `ruff` for lint and format.
- Module docstring per the template in `assets/templates/module_header.py`.
- No hidden globals. Configs are passed in, never read from a module-level path.
