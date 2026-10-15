# C4 Execution: Step-by-Step Build Guide

Owner: S A Gunerathne. Package: `execution/`.

In Phases 0 to 2 the simulator stands in for real execution, so C4's early work is the catalogue and the storage backends everyone uses, then the strategy executor. The live gateway is Phase 3.

## Phase 0

### C4-01 In-memory catalogue
- Files: `execution/catalogue.py`
- Build: `Catalogue` implementing `CatalogueAPI`, backed by a JSON file `data/clean/catalogue.json` of `ModelCard`s. `active_pool()` returns cards with status `warmup` or `active`.
- Test: register, retire, and pool membership behave as expected; cards round-trip.
- Commit: `feat(execution): in-memory model catalogue`

### C4-02 Session store (dict backend)
- Files: `execution/session.py`
- Build: `DictSessionStore` implementing `SessionStore`. Values: `W` (torch tensor), `trace` (list of `TraceItem`), `step_index`, `round`. Optional TTL check on read.
- Commit: `feat(execution): in-process session store`

### C4-03 Record store (Parquet backend)
- Files: `execution/records.py`
- Build: `ParquetRecordStore(dir)` buffering `DecisionRecord`, `ExecutionTrace`, `Score` rows and writing one Parquet file per table per flush. Nested fields stored as JSON strings.
- Test: write 1,000 records, read back, join decisions to scores on `(step_id, round)`.
- Commit: `feat(execution): Parquet record store`

## Phase 2

### C4-04 Provider adapter
- Files: `execution/providers.py`
- Build: `ProviderAdapter.call(card, messages, params) -> (text, usage, cost_usd, latency_ms, finish_reason)` via LiteLLM. Cost from usage times `ModelCard` prices. Retries with backoff on rate limits; timeouts recorded as errors, never raised to the framework.
- Test: with a mocked LiteLLM response, cost and latency fields are filled correctly.
- Commit: `feat(execution): provider adapter`

### C4-05 Strategy executor
- Files: `execution/executor.py`
- Build: `StrategyExecutor.run(strategy, request)` implementing `ExecutorAPI`. Mapping:

| Strategy field | Execution |
|---|---|
| `model_id` | provider model from the card |
| `tools` | request's `tools` filtered to the chosen subset |
| `memory = session` | prepend the episode's running summary; `retrieval` reserved |
| `depth` | per-provider mapping (reasoning effort or max output tokens) from `conf/depth_map.yaml` |
| `verification = self_check` | second call to the same model with a check prompt; sets `verifier_ok` |
| `verification = cross_model` | check call to the configured verifier model; its cost is charged |
| `topology = parallel` | 3 samples, select by verifier, else majority |
| `topology = sub_agents` | treated as `single` in v1 |
| `halt = escalate` | caller re-invokes the controller for round 1 |

- Produces `ExecutionTrace` and the `TraceItem` the controller sees (`verifier_ok` from verification, or from C3's live verifier when verification is `none`).
- Commit: `feat(execution): strategy executor`

### C4-06 Live environment
- Build: `LiveEnvironment` implementing `EnvironmentAPI` over a list of real tasks (e.g. a held-out sample of RouterBench prompts with references), using the executor and C3 scorer. `counterfactual` raises `NotImplementedError`.
- Why: the same evaluation code then runs on simulated and live episodes.
- Commit: `feat(execution): live environment`

### C4-07 Agree depth and verification mappings with C1
- Build: `conf/depth_map.yaml`, verifier model choice, parallel sample count. These define what the controller's Phase 2 heads mean in practice.
- Commit: `docs(execution): strategy execution mapping`

## Phase 3: live middleware

### C4-08 Gateway
- Files: `execution/gateway.py`
- Build: FastAPI app. `POST /v1/chat/completions` (OpenAI body). Headers in: `x-episode-id`, `x-step-type`, `x-budget-usd`, `x-cost-weight`, `x-episode-end`. Headers out: `x-controller-model`, `x-controller-version`, `x-episode-cost-usd`, `x-rounds`. `GET /v1/models` returns `orchestrator` plus the active pool.
- Flow per call: load session, build `TaskRep`, `controller.decide`, `executor.run`, `scorer.score`, `controller.observe`, optional escalation round, write records, return response.
- Test: integration test with a mocked provider from request to records.
- Commit: `feat(execution): OpenAI-compatible gateway`

### C4-09 Framework adapters
- Files: `execution/adapters/langgraph.py`, `crewai.py`, `agents_sdk.py`
- Build: hooks that set `x-episode-id` per workflow run, `x-step-type` where the node name allows, and `x-episode-end` on completion.
- Commit: `feat(execution): framework episode adapters`

### C4-10 Redis and PostgreSQL backends
- Files: `execution/session.py`, `execution/records.py`, `storage/migrations/001_init.sql`
- Build: `RedisSessionStore` (TTL 1 hour, `W` as float16 bytes), `PostgresRecordStore` (schema in `data-and-storage.md`). Same protocols, so nothing else changes.
- Commit: `feat(execution): Redis and PostgreSQL storage backends`

### C4-11 Onboarding pipeline
- Build: `POST /admin/models` registers a card, runs a smoke test, triggers C3's profiler, publishes with status `warmup`. `POST /admin/models/{id}/retire`.
- Commit: `feat(execution): model onboarding pipeline`

### C4-12 Load and latency test
- Build: controller overhead as a percentage of end-to-end time, p50 and p95, with a mocked provider at realistic latency.
- Commit: `test(execution): latency overhead`

## Hand-offs

| You provide | To | When |
|---|---|---|
| Catalogue, session store, record store | everyone | C4-01 to C4-03 |
| Executor and live environment | C3 sim-vs-live, final demo | C4-05, C4-06 |
| Gateway | demo | C4-08 |
