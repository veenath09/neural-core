# Contracts

The code is in `assets/common/contracts.py` and `assets/common/protocols.py`; copy them to `common/`. This file explains what each field means and who writes and reads it. If code and this file disagree, the code wins and this file must be fixed.

## Who writes and reads what

| Contract | Written by | Read by | Purpose |
|---|---|---|---|
| `ModelCard` | C4 registers; C3 fills `capability` | C1 model head, C4 executor | One candidate model, described by what it can do |
| `ToolCard` | C4 | C1 tool head (Phase 2) | One tool offered by the framework |
| `TraceItem` | C4 executor or C3 simulator | C1 (memory update, trace tokens) | What happened in one round, as the controller is allowed to see it |
| `TaskRep` | C4 or C3 simulator | C1 | Everything the controller sees when deciding |
| `Strategy` | C1 | C4 executor or C3 simulator | The decision |
| `DecisionRecord` | C1 | C2 | The decision plus what training needs |
| `ExecutionTrace` | C4 | C3 scorer, storage | What actually ran, with full usage and cost |
| `Score` | C3 | C2, storage, evaluation | Quality, cost and reward for one round |
| `EpisodeSkeleton` | C3 | C3 simulator, C2 rollouts | Policy-independent episode definition |
| `CheckpointMeta` | C2 | C1 loader, evaluation | Versioned trained weights |

## Field notes that matter

### TraceItem

- `verifier_ok` is a noisy correctness signal: in the simulator, true correctness flipped with probability 0.15 (seeded). It is the only correctness information the controller ever sees. Never add a field carrying ground truth.
- `round` is 0 for the first attempt at a step and 1 for the escalation.
- `dt` is 1.0 in the simulator. It exists for the CfC gate's time input. A `gate_notime` ablation is only meaningful once `dt` varies (live latency).

### TaskRep

- `trace` holds the last `k = 2` items, oldest first, including round 0 of the current step when building the round 1 decision.
- Embeddings are referenced by key (`task_emb_key`, `step_emb_key`); the controller's `TextEmbedder` resolves keys through its cache. This keeps contracts small and serialisable.
- `lam` is the cost weight. `None` during supervised training; sampled per episode in RL; set by the operator when serving.
- `expected_steps` is the episode length in simulation; in live serving it is an estimate (default 4).

### Strategy

- In Phases 0 and 1 only `model_id` and `halt` are decided; the other fields stay at defaults and the executor must ignore them.
- `confidence` is the calibrated success probability of the chosen model, not a separate head.
- `topology = "sub_agents"` is reserved; executors treat it as `single`.

### DecisionRecord

- `log_probs` and `entropy` have one entry per active head (`model`, `halt`, later others). In supervised greedy mode they are still filled, from the same distributions RL would sample.
- `complexity_band` = number of pool models predicted to fail, rounded to 0..4. C2 uses it to bucket the RL baseline.
- `decision_state` is the final `z_H` (length d). Stored for analysis and the optional value head.
- Rewards are not stored here. Join `Score` on `(step_id, round)`.

### Score

- `quality` uses ground truth. `verifier_ok` is copied from the trace for analysis only.
- `reward = quality - lam * cost_usd / cost_ref - mu * latency_ms / latency_ref - penalty`. `mu = 0` in v1.

### EpisodeSkeleton

- Stores only what does not depend on the policy: type, ordered prompt ids, budget, split, seed.
- Retries and propagation effects are produced live by the simulator because they depend on which models were chosen.

## Protocol rules

- `EnvironmentAPI` is implemented by both the simulator and the live executor. Training and evaluation code may only call `reset`, `step` and (simulator only) `counterfactual`.
- `counterfactual` returns, for every pool model, whether it would succeed on the current step given the history so far. It is how supervised labels are made. The live executor raises `NotImplementedError`.
- `SessionStore` and `RecordStore` hide storage. Phases 0 to 2 use the dict and Parquet backends.

## Versioning

- `CONTRACT_VERSION = "v1"` in `contracts.py`. `from_dict` rejects other versions and unknown fields, so mismatches fail loudly at load time instead of silently later.
- To change a contract: branch `contract/<change>`, edit the dataclass, bump `CONTRACT_VERSION`, update this file and `SKILL.md` Section 4 if relevant, get all members' approval.
- Adding an optional field with a default is still a version bump. Old files will fail to load, which is intended: regenerate them.
