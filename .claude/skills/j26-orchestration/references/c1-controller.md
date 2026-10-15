# C1 Controller: Step-by-Step Build Guide

Owner: V M Wickramaarachchi. Package: `controller/`. Read `architecture.md` Sections 4 to 8 first.

Each task lists files, what to build, the test that proves it is done, and the commit. Do them in order: each one is the base for the next and leaves something that runs.

## Phase 0: scaffold

### C1-01 Stub controller
- Files: `controller/stub.py`, `tests/controller/test_stub.py`
- Build: `StubController` implementing `ControllerAPI`. `decide` picks a random active model, `halt="stop"`, fills `DecisionRecord` with uniform log-probs and a zero `decision_state` of length 128.
- Test: output round-trips through `to_json`/`from_json`; `model_id` is in the pool.
- Why first: C2, C3 and C4 can build against it on day 1.
- Commit: `feat(controller): stub controller returning valid Strategy`

### C1-02 Text embedder with cache
- Files: `controller/embed.py`, `tests/controller/test_embed.py`
- Build: `TextEmbedder(model_name, cache_dir)`. `embed(texts) -> np.ndarray [n,384]`; `get(key) -> [384]`. Cache to `data/cache/emb_<model>.npy` plus a key index. Batch size 256, normalised embeddings.
- Test (fixture sample, CPU): shape `[n,384]`; a second call reads from cache without loading the model.
- Commit: `feat(controller): cached bge-small text embedder`

## Phase 1: prototype

### C1-03 Numeric features and StepEncoder
- Files: `controller/encoder.py`, `tests/controller/test_shapes.py`
- Build: `NumericFeaturizer.fit(train_rows)` saves mean/std to `norm_stats.json`. `StepEncoder` per the module table.

```python
class StepEncoder(nn.Module):
    """
    Purpose:  Turn a batch of TaskReps into H tokens (task, budget) and L tokens (step, trace).
    Inputs:   task_emb [B,e], budget_feats [B,5], step_type [B], step_emb [B,e],
              trace_emb [B,k,e], trace_feats [B,k,6], trace_model [B,k], trace_mask [B,k] (1 = real)
    Outputs:  h_tokens [B,2,d], l_tokens [B,1+k,d], obs [B,d] (newest real trace item)
    Notes:    Planner reads the task, fast module reads the execution: that split is
              what makes the specialisation question testable. Missing trace slots use
              a learned pad token. context_free masks earlier steps' trace items, the
              task token, the budget token and the memory readout, but keeps the
              current step's own round-1 item (C1-10).
    """
```

- budget_feats: `[budget_left/budget_total, log1p(spent), step_index/expected_steps, round, lam or 0]`.
- trace_feats: `[verifier_ok, log1p(cost), log1p(latency), log1p(tokens_out), escalated, tool_error]`.
- Test: exact output shapes at B=3; padded trace slots do not change `h_tokens`.
- Commit: `feat(controller): numeric featurizer and step encoder`

### C1-04 Blocks and the H/L core with plain L
- Files: `controller/blocks.py`, `controller/core.py`, `LICENSE-HRM-Text`
- Build: `RMSNorm`, `MLPBlock`, `HModule`, `LModule`, `HRMCore` with `memory: none`. Schedule per `architecture.md` Section 6. Header per `assets/templates/adapted_header.py` naming `sapientinc/HRM-Text`, `models/baselines/hrm_nocarry_bp_warmup.py`, Apache-2.0, and listing changes (MLP blocks instead of Transformer stacks, z_H seeded from task tokens, memory readout added).
- Core per D11: each module = 2 pre-norm SwiGLU MLP blocks, γ-free RMSNorm before each block and at the module exit (MagicNorm). `z_L` starts from a fixed buffer (truncated normal, std 1); `z_H` from `h = RMSNorm(mean(H tokens))`. L-step `z_L <- L(z_L + z_H + x)`; H-step `z_H <- H(z_H + z_L + h)` (re-injecting `h` is our deviation: under truncated backprop it is the only path by which the task and budget encoders get gradient).
- Truncated backprop: HRM-Text staircase. Only the last `bp` module calls keep gradients, `bp = 2 + floor(min(1, step / (0.2 total)) * 3)` (2 to 5 over the first 20% of training, passed in by the trainer), split `H_bp = min(2, bp - 1)`, `L_bp = bp - H_bp`; earlier calls run without grad.
- Test: shapes; with `n=2`, gradients reach only the last 2 inner steps (check `requires_grad` on intermediate states).
- Commit: `feat(controller): H/L core with truncated backprop, adapted from HRM-Text`

### C1-05 Heads
- Files: `controller/heads.py`
- Build: `ModelScoringHead`, `HaltingHead`, `ComplexityHead`.

```python
class ModelScoringHead(nn.Module):
    """
    Purpose:  Score a variable-length pool by matching what the step needs against
              what each model can do. No fixed per-model output slots, so a new
              ModelCard can be scored without retraining.
    Inputs:   z_H [B,d], caps [P,F] (per-step-type accuracy, log price, log latency),
              log_price [P], lam [B] (0 when None)
    Outputs:  success_logits [B,P] (trained with BCE on counterfactual success),
              policy_logits [B,P] (success minus learned price weight * log_price; used by RL)
    """
```

- Test: works for P=3 and P=5 with the same weights.
- Commit: `feat(controller): pool-scoring, halting and complexity heads`

### C1-06 Controller assembly and selection
- Files: `controller/model.py`, `controller/calibrate.py`
- Build: `Controller(cfg)` with `forward_batch(...)` for training and `decide`/`observe`/`reset` for the `ControllerAPI`. Selection rules per `architecture.md` Section 7. `Calibrator.fit(val_logits, val_labels)` by minimising NLL over one temperature.
- Test: `decide` returns a valid `Strategy` for all three modes; printed trainable parameter count.
- Commit: `feat(controller): controller assembly, selection rules, calibration`

### C1-07 Residual diagnostic
- Files: `controller/core.py` (return per-step deltas when `diagnostics=True`), `scripts/plot_residuals.py`
- Build: record `|z_L(t)-z_L(t-1)|` and `|z_H(c)-z_H(c-1)|`; plot mean over 1,000 val decisions.
- Done when: the plot shows L converging within each cycle and jumping when z_H changes. If L is flat after cycle 1, fix this before adding memory.
- Commit: `feat(controller): residual diagnostic`

### C1-08 TTT-Linear memory with constant rate (`gate_off`)
- Files: `controller/memory.py`

```python
class TTTLinearMemory(nn.Module):
    """
    Purpose:  Episode-level fast-weight memory. W is a small linear model trained
              online on the episode's own trace (self-supervised reconstruction).
    Inputs:   obs o [B,d] (newest round's trace token), W [B,d,d], gate state, dt
    Outputs:  W_new [B,d,d], readout m(h) = RMSNorm(W_new q) [B,d], surprise, rate, alpha
    Notes:    One write per new round outcome, not per decision or inner step: the
              episode, not the 6-step inner loop, is the sequence TTT learns from.
              One explicit delta-rule step (TTT-Linear inner model) with Titans-style
              forgetting, no momentum. W resets to W0 each episode. Unit-norm k makes
              each write a contraction, so no clipping is needed.
    """
```

Equations (D11):

```text
k = norm(W_k o)        v = W_v o        q = norm(W_q h)
s = v - W k                                      (surprise)
u = [s/|s|, log|s|, o]
r = gate(u, dt)        in (0, 1)
a = sigmoid(w_a . [log|s|, o] + b_a),  b_a = -4  (forgetting, starts near 0)
W <- W0 + (1 - a)(W - W0) + r * s k^T            (W0 learned, zero-initialised)
readout m = RMSNorm(W q)
dt = 1 / n_steps per write
```

- Write timing: the post-round pass writes that round; a round-2 outcome is written at the next step's first pass; the episode's first pass writes nothing.

- Test: `W` changes after `observe`, is restored by `reset`; with rate 0, `W` never changes.
- Commit: `feat(controller): TTT-Linear episode memory`

### C1-09 Gates
- Files: `controller/memory.py`
- Build: `CfCGate`, `StatelessGate`, `ConstantGate`, selected by `cfg.gate` (`cfc` | `stateless` | `const`). Log `surprise_norm`, `gate_rate` and `alpha` to `diagnostics`.
  - `CfCGate` (`full`): CfC cell, ncps default mode re-implemented: 16 units, backbone 32, LeCun-tanh, state reset per episode and carried across writes, time input `dt`; `r = sigmoid(w_r . c + b_r)`.
  - `StatelessGate` (`gate_stateless`): `r = sigmoid(MLP(u))` on the same input `u`, no state, no time.
  - `ConstantGate` (`gate_off`): one learned rate.
- The "liquid" claim needs `full` > `gate_stateless`.
- Test: CfC rate in (0,1), varies with surprise; constant gate ignores input.
- Commit: `feat(controller): CfC, stateless and constant gates`

### C1-10 Context-free masking
- Files: `controller/encoder.py`, `controller/model.py`, `tests/controller/test_masking.py`
- Build: `context_free` masks earlier steps' trace items, the task token, the budget token and the memory readout, but keeps the current step's own round-1 item (so within-step escalation stays possible and the random-chain control stays fair). `context_free_attn` applies the same rule through the attention key mask.
- Test: with `context_free`, outputs are exactly invariant when the masked inputs change; with `full` they differ.
- Commit: `feat(controller): context-free masking with invariance test`

### C1-11 Latency measurement
- Files: `scripts/measure_latency.py`
- Build: time `decide` on CPU at B=1, with and without cached task embedding, p50 and p95 over 1,000 calls.
- Measured so far (batch 1, laptop CPU, 1 thread, cached embedding; single seed, indicative): `full` 1.3 ms, `context_free` 1.0 ms, `full_attn` 3.2 ms, `context_free_attn` 2.9 ms; embedding a new prompt with bge-small about 52 ms.
- Commit: `feat(controller): latency measurement`

### Configs

Configs are Python presets in `controller/model.py` `CONFIGS`: `full`, `gate_off`, `gate_stateless`, `memory_off`, `context_free`, `full_attn`, `memory_off_attn`, `context_free_attn`. The YAML files in `assets/conf/experiment/` are the Hydra mirror for teammates. Measured trainable parameters: `full` 737,046; `gate_off` 726,150; `gate_stateless` 734,438; `memory_off` / `context_free` 660,483; `full_attn` 1,065,494; `memory_off_attn` / `context_free_attn` 988,931.

### Gated-attention variant (D12)

- Files: `controller/attention.py`, `controller/core.py` (`arch="attn"`), `controller/model.py`
- Built: six role tokens `[task, budget, step, trace-1, trace-2, memory]` with learned slot embeddings (no RoPE: roles, not positions). Each H/L module = 2 blocks of HRM-Text gated attention (fused bias-free gate/q/k/v projection, `o_proj(sigmoid(gate) * attn)`, 4 heads x 32) then SwiGLU, pre-norm, capped by a weightless RMSNorm (MagicNorm). Same H/L loop, fixed `z_L` init, input re-injection and gradient horizon as the MLP core, applied per token. An attention key mask hides absent slots and everything `context_free` must not see. H and L both receive the full masked token sequence; the heads read the step token. Residual diagnostics average valid tokens only.
- Configs: `full_attn`, `memory_off_attn`, `context_free_attn`. Compare within the family; the attn family has about 45% more parameters and about 2.5x the latency of the MLP family.

## Phase 2: full heads

### C1-12 Tool head and categorical heads
- Build: `ToolHead` (matches `z_H` against tool-description embeddings, variable U), `CategoricalHead` for memory, depth, verification, topology. Add to `log_probs`, `entropy`.
- Needs: C4 to execute these fields (C4-07). Agree the mapping first.
- Commit: `feat(controller): tool and categorical strategy heads`

### C1-13 Two-value halting
- Build: halting logits interpreted as value of stopping vs value of another round net of its cost; trained by C2's RL. Keep the BCE target for supervised warm start.
- Commit: `feat(controller): cost-aware two-value halting`

### C1-14 Calibration under shift
- Build: fit temperature on val, evaluate ECE on a shifted test mix (different source proportions). Provide `calibrate.py --shift` entry.
- Commit: `feat(controller): calibration under workload shift`

### C1-15 Specialisation probe
- Build: linear probes on `z_H` and `z_L` predicting task-level vs execution-level variables; compare with an untrained controller of the same shape as control.
- Commit: `feat(controller): H/L specialisation probe`

## Hand-offs

| You provide | To | When |
|---|---|---|
| `StubController` | everyone | C1-01 |
| `Controller.forward_batch` returning losses' inputs (success logits, halt logits, complexity, log-probs) | C2 | C1-06 |
| `decide`/`observe`/`reset` | C3 simulator driver, C4 | C1-06 |
| Diagnostics in `DecisionRecord` | C3 evaluation plots | C1-09 |
