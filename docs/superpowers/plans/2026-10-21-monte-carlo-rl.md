# Monte Carlo RL, Part 1: Measurement Reward and Simulator RL — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Role:** This is the warm start for **live online RL (Part 2), the main deliverable**. It produces the checkpoint, baseline and reward code that Part 2 continues from.

**Goal:** Fine-tune the supervised controller with Monte Carlo policy gradient in the simulator, so it optimises a whole episode's `quality − λ·cost` for any λ given at run time. The reward and the λ-oracle come from the `measurement/` package, and the complexity head stays trained and drives the policy. The result is the checkpoint and baseline that Part 2 (live online RL) starts from.

**Architecture:** Measurement comes first. One reward module in `measurement/` defines the reward, and the λ-oracle reuses `measurement.oracle.weighted_episode_oracle`. The controller gets λ- and complexity-conditioned policy outputs:
- `policy_logits = success_logits − λ·softplus(g(z_H, ln λ, c))·log_price`
- `halt_policy = halt_logit + h(z_H, ln λ, c)`, where `c = σ(complexity_logit)`

λ enters only the heads, so `context_free`, which masks the budget token, still sees it. The shared rollout gains `sample` and `serve` modes that record log-probs, entropy, complexity and per-decision rewards that sum exactly to the episode reward. The loss is policy gradient with a bucketed baseline plus the complexity auxiliary loss. A safety guard rolls back on any validation drop.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pandas, matplotlib, pytest.

**Spec:**
- `.claude/skills/j26-orchestration/references/c2-learning.md`: C2-05 baseline, C2-06 Monte Carlo PG, C2-07 safety guard
- `.claude/skills/j26-orchestration/references/contracts.md`: Score reward, `complexity_band`
- `docs/IMPLEMENTATION_PLAN_neural_core.md`: "Stage B: RL fine-tune" and line 91 (λ-oracle)

`RL_DESIGN.md`, which the implementation plan cites, doesn't exist; Task 2 writes the design in as D13. Part 2 is `docs/superpowers/plans/2026-10-21-online-rl.md`.

## Ownership and order

| Task | What | Component | Owner |
|---|---|---|---|
| 1 | RL reward and λ-oracle on `measurement/` | C3 | K V A Bethmi |
| 2, 3 | Policy outputs, distributions | C1 | V M Wickramaarachchi |
| 4, 5, 6 | Rollout, update, loop | C2 | R R H J Rathnayaka |
| 7 | Frontier evaluation | C3 | K V A Bethmi |
| 8 | Seed-0 simulator pilot | whoever runs it | |

Order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8. Task 1 comes first because Tasks 4, 5 and 7 test against it.

## Global Constraints

- Work on `main`, one commit per task. **Commit messages carry no co-author trailer.**
- Run from repo root with `.venv/bin/python`. Tests: `.venv/bin/python -m pytest -q`.
- The supervised path must stay unchanged in behaviour: with `lam=None` the controller's existing outputs are bit-identical, and every existing test passes **without edits**.
- λ grid: `LAMS = (0.1, 0.3, 0.6, 1.0, 2.0)`, one λ sampled uniformly per episode.
- `cost_ref = 0.001756663098` USD, read only through `CompositeScorer.from_yaml("conf/score.yaml").cfg.cost_ref_usd`.
- **Episode reward:** `R = quality − λ · (spent / n_steps) / cost_ref`, with `quality = n_correct / n_steps`. This equals `CompositeScorer.score(quality=q, cost_usd=spent/n_steps, lam=λ).reward`. Cost is the **per-step mean** because `cost_ref` is a per-step statistic (`mean_per_step_cost`). `mu = 0`, `penalty = 0`.
- **λ-oracle:** `weighted_episode_oracle(ep, lam=λ / (n_steps · cost_ref))`. Its `objective` then equals `R`, and `measurement/oracle.py` isn't changed.
- **Per-decision reward:** `r_t = Δn_correct_t / n_steps − λ · Δspent_t / (n_steps · cost_ref)`, where the Δ terms come from the decision's own transition. The `Δn_correct` from an automatic `end_step` after round 2 goes to that episode's last decision. `Σ_t r_t = R` exactly.
- Returns: undiscounted reward-to-go, `G_t = Σ_{k≥t} r_k`.
- **The complexity head has three jobs:**
  1. Auxiliary loss `0.2 · MSE(σ(complexity_logit), band_target)` on every valid decision. `band_target` is `features.target(...).band`, the exact fraction of the pool that fails.
  2. Baseline bucket `band = round(σ(complexity_logit) · 4)`, always 0..4 (contract `complexity_band`). It's fixed, so the baseline table keeps its shape when the pool grows (Part 2).
  3. Input to the policy adapter as `c = σ(complexity_logit)`, **not detached**, so the reward shapes it too.
- Baseline: table `b[band, λ index]`, zeros at the start, EMA `α = 0.05` with the mean return of each bucket in each batch. It's saved with the checkpoint, because Part 2 continues it.
- Advantage: `A = G − b`, normalised over choice slots: `(A − mean) / (std + 1e-8)` with the population std.
- Loss: `−mean(A.detach() · log π) − β · mean(H)` over choice slots, `+ 0.2 · complexity MSE` over valid slots. `β` decays linearly from `0.01` to `0.001`.
- Optimiser: AdamW, `lr = 1e-4`, weight decay `0.01`, gradient clip `1.0`, `bp_steps = 5`, 256 episodes per update, default 1,000 updates. RL starts from the supervised **EMA** `model.pt`.
- Safety guard: every 250 updates, evaluate val quality at each λ in serve mode. If any λ is more than `0.05` below the supervised reference, restore the last good weights and baseline, and halve the lr.
- `PRICE_INIT = 0.1`: at initialisation `softplus(g) = 0.1` and the halt shift is 0.
- `log_price[m] = ln(card_cost[m] / cost_ref)`, from `World.card_costs`.

## Review Focus

1. **No model is affordable at CHOOSE:** the rollout falls back to all models, `play` ends the episode without charging, and the rewards must still sum to `R`. Test in Task 4.
2. **A forced stop** (HALT with no stronger affordable model) must not count as a choice, but its step reward stays in the returns. Test in Task 4.
3. **Old supervised checkpoints** lack `heads.lam_adj.*`. Loading accepts exactly those missing keys and rejects anything else. Test in Task 6.
4. **`context_free` sees λ but not context:** its policy outputs are invariant to masked inputs and change with λ. Test in Task 2.
5. **One choice slot, or equal advantages,** must not give a NaN loss. Test in Task 5.

---

### Task 1: RL reward and λ-oracle on the measurement package (C3)

**Files:**
- Create: `measurement/rl_reward.py`
- Test: `tests/test_measurement_rl_reward.py`

**Interfaces:**
- Consumes: `CompositeScorer` (`measurement/scorer.py`), `weighted_episode_oracle` and `WeightedEpisodeOracleResult` (`measurement/oracle.py`).
- Produces:
  - `load_cost_ref(path: str | Path = "conf/score.yaml") -> float`
  - `episode_reward(scorer: CompositeScorer, quality: float, spent: float, n_steps: int, lam: float) -> float`, implemented as `scorer.score(quality=quality, cost_usd=spent / n_steps, lam=lam).reward`
  - `step_reward(d_quality: float, d_spent: float, n_steps: int, lam: float, cost_ref: float) -> float`. `d_quality` is the step's correctness: an exact 0/1 in the simulator, a verifier `Score.quality` in [0, 1] live (Part 2)
  - `lam_oracle(ep: Episode, lam: float, cost_ref: float) -> WeightedEpisodeOracleResult`, implemented as `weighted_episode_oracle(ep, lam / (ep.n_steps * cost_ref))`
  - `lam_oracle_points(skeletons, table, model_ids, cost_ref: float, lams=LAMS, cache_dir: Path | None = None) -> pd.DataFrame` with columns `lam`, `quality`, `cost_usd` and `reward` (means over episodes). It's cached as `cache_dir/lam_oracle_<blake2b of episode ids + lams>.json`. `reward` is recomputed with `episode_reward`, not taken from `objective`.
  - `LAMS = (0.1, 0.3, 0.6, 1.0, 2.0)` lives here; the controller imports it from here.

- [ ] **Step 1: Write the failing tests**, using `episode` from `tests/test_env.py` (costs `[1, 2, 4, 8]`):
  - `test_load_cost_ref_reads_conf`: the result equals `0.001756663098`.
  - `test_episode_reward_uses_per_step_cost`: with `cost_ref=1`, `quality=0.5`, `spent=6`, `n_steps=3` and `lam=0.5`, the result is `0.5 − 0.5·2 = −0.5`.
  - `test_step_rewards_sum_to_episode_reward`: for a hand-built sequence of `(d_correct, d_spent)` pairs, the sum of `step_reward` equals `episode_reward`.
  - `test_lam_oracle_objective_equals_reward`: on `episode("random", [[0, 0, 0, 1], [1, 1, 1, 1]])`, for each λ in `LAMS`, `lam_oracle(...).objective ≈ episode_reward(quality, cost_usd, n_steps, λ)`.
  - `test_lam_oracle_trades_quality_for_cost`: on a one-step `episode("random", [[0, 0, 0, 1]])` with `cost_ref=1`: at `lam=0.01` it gets `quality == 1` with `cost == 8`; at `lam=1.0` it gets `quality == 0` with `cost == 1`.
  - `test_lam_oracle_points_shape_and_cache`: 5 rows; the second call reads the cache file.

- [ ] **Step 2: Run the tests and confirm they fail.** Run `.venv/bin/python -m pytest -q tests/test_measurement_rl_reward.py`. Expected: FAIL with ImportError.
- [ ] **Step 3: Implement** the Interfaces above.
- [ ] **Step 4: Run the tests.** Run `.venv/bin/python -m pytest -q tests/test_measurement_rl_reward.py tests/test_measurement_weighted_oracle.py`. Expected: PASS, and the weighted-oracle tests are unedited.
- [ ] **Step 5: Commit.** `git commit -m "feat(measurement): RL reward and lambda-oracle on the composite scorer"`

---

### Task 2: Spec D13 and λ/complexity-conditioned policy outputs (C1)

**Files:**
- Modify: `docs/IMPLEMENTATION_PLAN_neural_core.md` (insert `### D13` immediately after `### D12`)
- Modify: `controller/heads.py`
- Modify: `controller/model.py` (`Output`, `Controller.forward`, `Controller._forward_tokens`)
- Create: `controller/policy.py` (constants and `log_price` only in this task)
- Test: `tests/test_policy_heads.py`

**Interfaces:**
- Consumes: `LAMS` from `measurement/rl_reward.py` (Task 1).
- Produces:
  - `controller/policy.py`: `PRICE_INIT = 0.1` and `log_price(card_costs: np.ndarray, cost_ref: float) -> torch.Tensor` returning `[M]` float32.
  - `Heads.lam_adj = nn.Linear(d + 2, 2)`, with weights zero and bias `[ln(expm1(PRICE_INIT)), 0.0]`. Its input is `cat([z_h, ln(lam)[:, None], σ(complexity_logit)[:, None]])`.
  - `Heads.policy(z_h [B,d], model_logits [B,M], halt_logit [B], complexity_logit [B], lam [B], log_price [M]) -> tuple[policy_logits [B,M], halt_policy_logit [B]]`:
    - `raw = lam_adj(...)`
    - `policy = model_logits − (lam · softplus(raw[:, 0]))[:, None] · log_price[None]`
    - `halt = halt_logit + raw[:, 1]`
  - `Output` gains `policy_logits: torch.Tensor | None = None` and `halt_policy_logit: torch.Tensor | None = None`.
  - `Controller.forward(b, memory=None, bp_steps=5, record=False, lam: torch.Tensor | None = None, log_price: torch.Tensor | None = None) -> Output`. The new fields are filled only when both are given, in both the `mlp` and `attn` paths.

- [ ] **Step 1: Add D13 to the spec:**

```markdown
### D13. Monte Carlo RL (Stage B, simulator; Stage C, live online)

Stage B starts from the supervised EMA checkpoint. Reward per episode `R = quality − λ·(spent/n_steps)/cost_ref` (`CompositeScorer` with `cost_usd = spent/n_steps`; `cost_ref` from `conf/score.yaml` is a per-step statistic), paid out per decision so that `Σ r_t = R`. λ ∈ {0.1, 0.3, 0.6, 1.0, 2.0}, sampled per episode. λ enters only the heads: `policy_logits = success_logits − λ·softplus(g(z_H, ln λ, c))·ln(card_cost/cost_ref)`, `halt_policy = halt_logit + h(z_H, ln λ, c)`, `c = σ(complexity_logit)`. It never enters the tokens, because `context_free` masks the budget token and must still see λ. The complexity head stays trained (auxiliary 0.2·MSE to the exact band), keys the bucketed baseline `b[band, λ]`, and conditions the policy. Loss `−mean(A·log π) − β·mean(H) + 0.2·MSE_complexity`, β 0.01 → 0.001, lr 1e-4, clip 1.0. Safety guard every 250 updates: val quality more than 5 points below the supervised reference at any λ → restore the last good weights, halve the lr. Frontier: serve mode per λ against the λ-oracle `weighted_episode_oracle(ep, λ/(n_steps·cost_ref))`. Configs `full` and `context_free` × 3 seeds. Stage C (live online RL) continues from the Stage B checkpoint and baseline; see the online RL plan.
```

- [ ] **Step 2: Write the failing tests** in `tests/test_policy_heads.py`. Reuse `make_model`, `sample_batch` and `perturb_context` from `tests/test_model.py`; `LP = torch.log(torch.tensor([1.0, 2.0, 4.0, 8.0]))`.

```python
def test_no_lam_means_no_policy_outputs():
    m = make_model("full")
    out = m(sample_batch(), m.init_memory(3))
    assert out.policy_logits is None and out.halt_policy_logit is None

def test_policy_at_init_is_success_minus_lam_times_price_init():
    m = make_model("full"); lam = torch.full((3,), 0.6)
    out = m(sample_batch(), m.init_memory(3), lam=lam, log_price=LP)
    assert torch.allclose(out.policy_logits, out.model_logits - 0.6 * 0.1 * LP[None], atol=1e-6)
    assert torch.allclose(out.halt_policy_logit, out.halt_logit)

def test_higher_lam_prefers_cheaper_models():
    m = make_model("full"); b = sample_batch()
    lo = m(b, m.init_memory(3), lam=torch.full((3,), 0.1), log_price=LP).policy_logits.softmax(-1)
    hi = m(b, m.init_memory(3), lam=torch.full((3,), 2.0), log_price=LP).policy_logits.softmax(-1)
    assert (hi[:, 0] > lo[:, 0]).all() and (hi[:, 3] < lo[:, 3]).all()

def test_policy_gradient_reaches_complexity_head():
    m = make_model("full")
    with torch.no_grad():
        m.heads.lam_adj.weight[:, -1] = 1.0
    out = m(sample_batch(), m.init_memory(3), lam=torch.full((3,), 1.0), log_price=LP)
    out.policy_logits.sum().backward()
    assert m.heads.complexity.weight.grad.abs().sum() > 0

@pytest.mark.parametrize("name", ["context_free", "context_free_attn"])
def test_context_free_policy_ignores_context_but_sees_lam(name):
    m = make_model(name); b = sample_batch(); lam = torch.full((3,), 1.0)
    a = m(b, m.init_memory(3), lam=lam, log_price=LP)
    c = m(perturb_context(b), m.init_memory(3), lam=lam, log_price=LP)
    assert torch.equal(a.policy_logits, c.policy_logits) and torch.equal(a.halt_policy_logit, c.halt_policy_logit)
    with torch.no_grad():
        m.heads.lam_adj.weight[:, -2] = 1.0   # the ln λ column
    d = m(b, m.init_memory(3), lam=torch.full((3,), 2.0), log_price=LP)
    assert not torch.equal(a.policy_logits, d.policy_logits)

def test_log_price_is_natural_log_relative_to_cost_ref():
    assert torch.allclose(log_price(np.array([1e-3, 2e-3]), 1e-3), torch.tensor([0.0, math.log(2)]))
```

- [ ] **Step 3: Run the tests and confirm they fail.** Run `.venv/bin/python -m pytest -q tests/test_policy_heads.py`. Expected: FAIL (`unexpected keyword argument 'lam'`).
- [ ] **Step 4: Implement** the Interfaces above.
- [ ] **Step 5: Run the full suite.** Run `.venv/bin/python -m pytest -q`. Expected: all pass, with 262 more parameters.
- [ ] **Step 6: Commit.** `git commit -m "feat(controller): lambda- and complexity-conditioned policy outputs (D13)"`

---

### Task 3: Policy distributions and selection modes (C1)

**Files:**
- Modify: `controller/policy.py`
- Test: `tests/test_policy.py`

**Interfaces:**
- Consumes: `Output.policy_logits` and `Output.halt_policy_logit` (Task 2).
- Produces:
  - `choose_step(policy_logits [B,M], allowed [B,M] bool, mode: str, gen: torch.Generator | None) -> tuple[model [B] long, logp [B], entropy [B]]`. `mode` is `"sample"` (`torch.multinomial(..., generator=gen)`) or `"serve"` (argmax). Disallowed logits are set to `−inf`. A row with nothing allowed raises `ValueError`.
  - `halt_step(halt_policy_logit [B], policy_logits [B,M], options [B,M] bool, mode: str, gen) -> tuple[model [B] long (−1 = stop), logp [B], entropy [B], has_choice [B] bool]`:
    - A row with no option is a forced stop: `logp = 0`, `entropy = 0`, `has_choice = False`.
    - Otherwise `p = σ(halt_policy_logit)`. Sample mode escalates iff `torch.rand(generator=gen) < p`; serve mode escalates iff `p > 0.5`.
    - Escalate: `m₂` comes from `choose_step` over `options`, and `logp = logσ(x) + ln π(m₂)`. Stop: `logp = logσ(−x)`.
    - `entropy = H_Bernoulli(p) + H_Categorical(options)`.
  - `logp` and `entropy` keep the autograd graph.

- [ ] **Step 1: Write the failing tests** in `tests/test_policy.py`:
  - `test_choose_never_picks_a_disallowed_model`: 200 samples with `allowed = [[F, T, F, T]]` give only models 1 and 3.
  - `test_choose_logp_matches_masked_softmax`: `logp` and `entropy` equal the masked `log_softmax` and the categorical entropy.
  - `test_serve_is_argmax_and_deterministic`
  - `test_halt_escalate_logp_is_sum_of_parts`: with logit `+20`, `logp ≈ ln σ(20) + ln π(m₂)`.
  - `test_halt_stop_logp`: with logit `−20`, `model == −1` and `logp ≈ ln σ(20)`.
  - `test_forced_stop_has_no_choice`: an all-False `options` row gives `−1`, `0`, `0` and `False`.
  - `test_logp_has_grad`: gradients reach a leaf `policy_logits`.
- [ ] **Step 2: Run the tests and confirm they fail.** Run `.venv/bin/python -m pytest -q tests/test_policy.py`. Expected: FAIL with ImportError.
- [ ] **Step 3: Implement** `choose_step` and `halt_step`.
- [ ] **Step 4: Run the tests.** Run `.venv/bin/python -m pytest -q tests/test_policy.py`. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(controller): sample and serve policies with log-probs and entropy"`

---

### Task 4: Sampling rollout with per-decision rewards (C2)

**Files:**
- Modify: `train/rollout.py`
- Test: `tests/test_rollout_rl.py`

**Interfaces:**
- Consumes: `choose_step`, `halt_step`, `log_price` (Tasks 2–3); `step_reward`, `LAMS` (Task 1); `features.target`.
- Produces:
  - `RLTrace`, padded to `T` = the most decisions any episode in the batch made:
    - `logp [n,T]`, `entropy [n,T]` and `complexity [n,T]` (`σ(complexity_logit)`), all with grad
    - `reward [n,T]` float32, `band [n,T]` long, `band_target [n,T]` float32
    - `lam_idx [n]` long
    - `choice [n,T]` bool and `valid [n,T]` bool. Padding is 0 / False.
  - `RolloutOutput.rl: RLTrace | None = None`.
  - `rollout(..., policy=...)` also accepts `"sample"` and `"serve"`, plus keyword arguments `lam_idx: np.ndarray | None = None` and `cost_ref: float | None = None`. Both are required in those modes, otherwise `ValueError`. `greedy` and `mixture` are unchanged.
  - The sampling `torch.Generator` is seeded once per rollout with `int(rng.integers(2**31))`.

Behaviour in `sample` and `serve` modes:
- Pass `lam = torch.tensor(LAMS)[lam_idx[active]]` and `log_price(world.card_costs, cost_ref)` to the model.
- CHOOSE: `allowed[k] = card_costs[k] ≤ budget_left`, falling back to all models if none qualify.
- HALT: the same `options` as the existing code.
- Rewards come from `step_reward`. `band` comes from the current `complexity_logit`; `band_target = target(ep, s, phase).band`.
- `valid[i].sum() == len(decision_actions[i])`.

- [ ] **Step 1: Write the failing tests** in `tests/test_rollout_rl.py`, reusing `world_and_episodes` and `tiny_model` from `tests/test_rollout.py`:
  - `test_sample_rollout_trace_is_aligned`: `valid[i].sum() == len(decision_actions[i])`, `choice ⊆ valid`, `T ≤ 12`.
  - `test_rewards_sum_to_measurement_reward`: per episode, `reward[i].sum() ≈ episode_reward(scorer, r.quality, r.cost_usd, n_steps_i, LAMS[lam_idx[i]])` (abs tol 1e-5).
  - `test_rewards_sum_when_nothing_is_affordable`: budget `0.25 × min card cost`, below every fixture row cost. The episode ends at the first CHOOSE, the sum is 0, and there's one valid slot.
  - `test_forced_stop_is_not_a_choice_but_keeps_its_reward`: the tight-budget skeleton from `test_no_affordable_stronger_model_means_stop`, in `serve` mode. Every HALT slot has `choice == False` and `valid == True`, and the sum still equals `episode_reward`.
  - `test_band_target_is_the_exact_fail_fraction`: every valid `band_target` equals `target(...).band` for that state, in `[0, 1]`.
  - `test_policy_gradient_reaches_memory_gate`: `rl.logp[rl.choice].sum().backward()` populates the gate readout gradient.
  - `test_serve_is_deterministic`
  - `test_sample_requires_lam_and_cost_ref`
- [ ] **Step 2: Run the tests and confirm they fail.** Run `.venv/bin/python -m pytest -q tests/test_rollout_rl.py`. Expected: FAIL.
- [ ] **Step 3: Implement** the `sample` and `serve` branches. Build per-episode lists and pad them into `RLTrace` at the end. Credit the automatic `end_step` after round 2 to the last decision.
- [ ] **Step 4: Run the full suite.** Run `.venv/bin/python -m pytest -q`. Expected: all pass, and `tests/test_rollout.py` is unedited.
- [ ] **Step 5: Commit.** `git commit -m "feat(learning): sampling and serve rollout with per-decision rewards"`

---

### Task 5: Bucketed baseline and the policy-gradient update (C2-05, C2-06)

**Files:**
- Create: `train/baseline.py`
- Create: `train/rl.py` (update math only in this task)
- Test: `tests/test_rl_update.py`

**Interfaces:**
- Consumes: `RLTrace` (Task 4).
- Produces:
  - `BucketedBaseline(n_bands: int = 5, n_lams: int = 5, alpha: float = 0.05)`:
    - `table [n_bands, n_lams]`, zeros at the start
    - `value(band, lam_idx) -> [N]`
    - `update(band, lam_idx, returns) -> None`: for each bucket present, `b ← b + α(mean − b)`
    - `state_dict() -> dict` and `load_state_dict(d) -> None`
  - `COMPLEXITY_AUX = 0.2`
  - `returns_to_go(reward [n,T]) -> [n,T]`
  - `entropy_beta(update: int, total: int, start=0.01, end=0.001) -> float` (linear, clamped)
  - `rl_loss(trace, baseline, beta) -> tuple[torch.Tensor, dict]`, implementing the Global Constraints loss including the complexity term
  - `rl_update(model, trace, baseline, opt, beta) -> dict` with keys `loss`, `pg_loss`, `complexity_mse`, `episode_return` (mean `G[:, 0]`), `entropy` and `grad_norm`. It computes advantages from the baseline before updating it, then updates the baseline with the choice-slot returns.
- [ ] **Step 1: Write the failing tests** in `tests/test_rl_update.py`:
  - `test_returns_to_go`: `[[1, 2, 3, 0]] → [[6, 5, 3, 0]]`.
  - `test_entropy_beta_endpoints_and_clamp`
  - `test_baseline_moves_five_percent_and_leaves_other_buckets`: after `update(band=[1, 1], lam=[2, 2], returns=[1, 3])`, `table[1, 2] == 0.1` and everything else is 0. `state_dict` round-trips.
  - `test_single_choice_slot_gives_finite_loss`: one choice slot, or equal advantages, gives a finite loss.
  - `test_complexity_term_pulls_toward_band_target`: one `rl_update` with zero advantages and a complexity far from its target moves `σ(complexity)` toward the target.
  - `test_positive_advantage_raises_logp`: on a hand-made one-parameter policy.
  - `test_policy_converges_on_easy_world` (spec C2-06): build `make_table` with `correct[:, 0] = True` and embeddings `torch.randn(n_prompts, 384)`. Train a tiny `full` model with `lr=3e-3` for 60 updates × 32 episodes. Then a `serve` rollout at λ index 3 picks model 0 in at least 90% of CHOOSE decisions. Under 60 s.
- [ ] **Step 2: Run the tests and confirm they fail.** Run `.venv/bin/python -m pytest -q tests/test_rl_update.py`. Expected: FAIL.
- [ ] **Step 3: Implement** the Interfaces above. Clip with `clip_grad_norm_(model.parameters(), 1.0)`.
- [ ] **Step 4: Run the tests.** Run `.venv/bin/python -m pytest -q tests/test_rl_update.py`. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(learning): bucketed baseline and Monte Carlo policy gradient with complexity auxiliary"`

---

### Task 6: Safety guard, simulator RL loop and CLI (C2-07)

**Files:**
- Create: `train/safety.py`
- Modify: `train/rl.py` (loop and CLI)
- Modify: `train/train.py` (`load_model`)
- Test: `tests/test_rl_train.py`

**Interfaces:**
- Consumes: Tasks 1–5; `load_data`, `evaluate` and `make_episodes` from `train/train.py`.
- Produces:
  - `SafetyGuard(reference_quality: float, tol: float = 0.05)` with `ok(quality_by_lam: dict[int, float]) -> bool`.
  - `load_model(run_dir, data=None, strict: bool = True)`. With `strict=False`, only missing `heads.lam_adj.*` keys are allowed; any other mismatch raises `RuntimeError`.
  - `evaluate_rl(model, skeletons, world, lam_idx: int, cost_ref: float, batch_size: int = 512) -> dict` with keys `quality`, `cost_usd`, `reward`, `escalation_rate`, `choice_share`, `complexity_mae` and `by_type` (per `ep_type`: quality, cost and reward). It uses the serve rollout under `no_grad` in eval mode.
  - `train_rl(config: str, seed: int, init_dir: Path, updates: int = 1000, batch_size: int = 256, lr: float = 1e-4, eval_every: int = 250, max_val: int | None = None, out_dir: Path | None = None, data: Data | None = None, reference_quality: float | None = None, log=print) -> dict`. `out_dir` defaults to `init_dir / "rl"`. `reference_quality` overrides the measured supervised reference (for tests).
  - Writes:
    - `out_dir/model.pt`: the best mean val reward among checks that pass the guard, or the supervised start if none pass
    - `out_dir/baseline.pt`: the baseline matching `model.pt`
    - `out_dir/metrics.json` with `config`, `seed`, `stage: "rl_sim"`, `init_dir`, `data_dir`, `lams`, `cost_ref`, `reference_quality` and `history`
  - CLI: `python -m train.rl --config full --seed 0 --init runs/routerbench/full-s0 [--updates 1000] [--batch-size 256] [--max-val N]`

Loop:
1. Load with `load_model(init_dir, data, strict=False)`.
2. The reference is supervised greedy val quality, using `tau`/`halt_tau` from `init_dir/tau.json` if present, otherwise 0.5/0.5.
3. Evaluate and log update 0.
4. Each update: draw `batch_size` train skeletons without replacement and `lam_idx = rng.integers(5, size=batch_size)`, run the `sample` rollout with `bp_steps=5`, then `rl_update` with `entropy_beta(u, updates)`.
5. Every 25 updates, log a JSON row: `update`, `train_return`, `pg_loss`, `complexity_mse`, `entropy`, `beta`, `lr`, `grad_norm`.
6. Every `eval_every` updates and at the end, log `val` per λ and `mean_val_reward`.
   - Guard passes: snapshot `last_good` (weights and baseline) and save if it's the best so far.
   - Guard fails: restore `last_good`, halve the lr, and add the alarm `"rollback"`.

- [ ] **Step 1: Write the failing tests** in `tests/test_rl_train.py`, using `tiny_data()` from `tests/test_train.py`:
  - `test_safety_guard`: `ok({0: 0.66, 1: 0.71})` with reference 0.7 is True; `ok({0: 0.64})` is False.
  - `test_load_old_checkpoint_without_lam_adj`: deleting the `heads.lam_adj.*` keys still loads with `strict=False`. Deleting `heads.halt.weight` raises. The missing `lam_adj` keys raise under `strict=True`.
  - `test_train_rl_smoke`: 1 supervised epoch, then `train_rl(..., updates=4, batch_size=8, eval_every=2)`. `rl/model.pt`, `rl/baseline.pt` and `rl/metrics.json` exist with `stage == "rl_sim"`, `load_model(run / "rl", data)` loads strictly, and every history value is finite.
  - `test_rollback_restores_weights_and_halves_lr`: with `reference_quality=1.0`, the lr halves at each check, every check row has `"rollback"` in `alarms`, and the final weights equal the supervised start.
- [ ] **Step 2: Run the tests and confirm they fail.** Run `.venv/bin/python -m pytest -q tests/test_rl_train.py`. Expected: FAIL.
- [ ] **Step 3: Implement** the Interfaces and loop above.
- [ ] **Step 4: Run the full suite.** Run `.venv/bin/python -m pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit.** `git commit -m "feat(learning): simulator RL loop with safety rollback"`

---

### Task 7: Quality–cost frontier evaluation (C3)

**Files:**
- Create: `eval/rl_frontier.py`
- Test: `tests/test_rl_frontier.py`

**Interfaces:**
- Consumes: `evaluate_rl` (Task 6); `lam_oracle_points` and `load_cost_ref` (Task 1); `init_dir/tau_sweep.csv` from `eval/tau_sweep.py`.
- Produces:
  - `frontier(rl_dir: Path, lams=LAMS, max_val: int | None = None) -> pd.DataFrame` with columns `source` (`rl`, `lam_oracle` or `supervised_tau`), `lam`, `tau`, `halt_tau`, `quality`, `cost_usd` and `reward`, NaN where a field doesn't apply. RL and oracle rows use the same val skeletons. Supervised rows come from `tau_sweep.csv` in the `init_dir` named by `rl_dir/metrics.json`, and are skipped if it's missing.
  - `by_type(rl_dir) -> pd.DataFrame` with quality, cost and reward per `ep_type` per λ, for the `full` vs `context_free` table.
  - Writes `rl_dir/rl_frontier.csv`, `rl_dir/rl_by_type.csv` and `rl_dir/rl_frontier.png` (x: cost per episode, y: quality, one series per source).
  - CLI: `python -m eval.rl_frontier <rl_dir> [more] [--max-val N]`
- [ ] **Step 1: Write the failing tests**, reusing the Task 6 smoke setup:
  - `test_frontier_rows_and_files`: 5 `rl` and 5 `lam_oracle` rows; all three files exist.
  - `test_oracle_bounds_rl_reward`: for every λ, `lam_oracle.reward ≥ rl.reward − 1e-9`. The oracle is an exact per-episode upper bound.
- [ ] **Step 2: Run the tests and confirm they fail.** Run `.venv/bin/python -m pytest -q tests/test_rl_frontier.py`. Expected: FAIL.
- [ ] **Step 3: Implement** `frontier`, `by_type` and the CLI.
- [ ] **Step 4: Run the full suite.** Run `.venv/bin/python -m pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit.** `git commit -m "feat(eval): RL quality-cost frontier against the lambda-oracle"`

---

### Task 8: Seed-0 simulator RL pilot

**Files:**
- Create: `runs/launch_rl_seed0.sh`
- Create or append: `docs/experiment_log.md`

**Prerequisites:** the supervised runs `runs/routerbench/{full,context_free}-s0` have finished, each with `tau.json` from `python -m eval.tau_sweep <run_dir>`.

- [ ] **Step 1: Write the launcher.** It runs `full` and `context_free` in parallel with `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`. Each one runs `.venv/bin/python -u -m train.rl --config <c> --seed 0 --init runs/routerbench/<c>-s0 > runs/routerbench/<c>-s0/rl.log 2>&1`, then `python -m eval.rl_frontier runs/routerbench/<c>-s0/rl`.
- [ ] **Step 2: Run it and check health.** Run `zsh runs/launch_rl_seed0.sh`. Healthy means:
  - every 25-update row is finite
  - entropy doesn't reach 0 within the first 100 updates
  - `complexity_mse` stays within 1.5× its update-0 value
  - at most one rollback per run
- [ ] **Step 3: Record the results** in `docs/experiment_log.md`, per config:
  - val reward per λ at update 0 versus the final checkpoint (**done when the final value is at least the update-0 value at every λ**)
  - quality and cost per λ, the λ-oracle per λ, `complexity_mae`, and the number of rollbacks
  - the `full` vs `context_free` per-type table, labelled "seed 0, indicative only"
- [ ] **Step 4: Commit.** `git add runs/launch_rl_seed0.sh docs/experiment_log.md && git commit -m "exp: seed-0 simulator RL pilot"`

Seeds 1 and 2 run after the analysis freeze (skill §12).

## Out of scope (in Part 2)

Live online RL, new-model onboarding with the exploration bonus, card updates from live outcomes, and the `Controller` session API. See `docs/superpowers/plans/2026-10-21-online-rl.md`. `train/online_learner.py` and `train/replay_buffer.py` are untouched here.
