# C2 Learning: Step-by-Step Build Guide

Owner: R R H J Rathnayaka. Package: `learning/`.

C2 trains the controller in two stages: a supervised warm start on oracle labels, then Monte Carlo RL that optimises the whole episode's quality minus cost.

## The one thing to get right

The controller's memory `W` carries across decisions within an episode. So **training batches are whole episodes, processed step by step in order**, with gradients flowing through `W` from later decisions back to earlier ones. Shuffling individual steps into a batch would cut the memory out of training and make `full` look the same as `memory_off`.

## Phase 0

### C2-01 Checkpoint manager
- Files: `learning/checkpoints.py`
- Build: save `state_dict`, `CheckpointMeta` as `meta.json`, `norm_stats.json`, resolved config, to `checkpoints/<checkpoint_id>/`. `load` refuses a different `contract_version`.
- Test: save, load, identical outputs on a fixed batch.
- Commit: `feat(learning): checkpoint manager`

## Phase 1: supervised

### C2-02 Rollout worker
- Files: `learning/rollout.py`
- Build: `RolloutWorker(env, controller, behaviour)` runs B episodes in lockstep through the `EnvironmentAPI`. Per visited round it records the `TaskRep`, the action taken, and labels from `env.counterfactual`:
  - `succeeds[m]` for every pool model,
  - halting label: 1 if the round failed and some strictly more expensive affordable model succeeds, else 0,
  - complexity label: difficulty band / 4 given history.
- Behaviour policy per step: per-step oracle 50%, uniform random 25%, current model 25% (oracle instead of the model in epoch 1); the random source escalates with probability 0.3. The exact episode oracle is the evaluation ceiling, not part of the mixture. Labels are for every model regardless of which one the behaviour policy picked.
- Episodes are regenerated each epoch from the same skeletons, so the current-model share tracks the model as it improves (limits exposure bias).
- Commit: `feat(learning): rollout worker with mixed behaviour policy and counterfactual labels`

### C2-03 Supervised trainer
- Files: `learning/supervised.py`, `scripts/train.py`

```python
class SupervisedTrainer:
    """
    Purpose:  Stage A. Fit success, halting and complexity heads on oracle labels.
    Inputs:   skeletons (train, val), env, controller, cfg (conf/train_supervised.yaml)
    Outputs:  CheckpointMeta (stage="supervised"), training curves
    Notes:    Loss = 1.0*BCE(success_logits, succeeds) + 0.5*BCE(halt, halt_label)
              + 0.2*MSE(complexity, band/4), averaged over real (unpadded) rounds.
              Episodes are unrolled in order so gradients pass through W.
              Passes the truncated-backprop horizon (2 -> 5 over first 20%) to the core.
              Early stop on val loss, patience 3. Fits the calibration temperature at the end.
              EMA of weights, warm-up decay min(0.999, (1+t)/(10+t)); validation
              and checkpoints use the EMA weights.
    """
```

- EMA warm-up: a fixed 0.999 left about 21% of the random init in the averaged weights after a 1,580-step run, so the decay warms up from (1+t)/(10+t).
- Loss weights: the spec is 1.0 / 0.5 / 0.2 (above); the current code (`train/rollout.py`) sums the three terms 1 / 1 / 1. Open item: fix before the analysis freeze.

- Test: loss decreases on the 200-prompt fixture in 2 epochs; checkpoint loads.
- Commit: `feat(learning): supervised trainer`

### C2-04 Config sweep runner
- Build: `scripts/train.py experiment=<name> seed=<s>` for `full`, `gate_off`, `gate_stateless`, `memory_off`, `context_free`, and the attention family `full_attn`, `memory_off_attn`, `context_free_attn`; seeds 0, 1, 2.
- Commit: `feat(learning): experiment sweep runner`

## Phase 1: RL

### C2-05 Bucketed baseline
- Files: `learning/baseline.py`
- Build: `BucketedBaseline(alpha=0.05)`, table indexed by `(complexity_band, lam index)`, initialised to 0, EMA update with the mean return of each bucket per batch.
- Why: hard tasks earn less than easy ones whatever the policy does; bucketing stops the policy being blamed for task difficulty. It works from single episodes, which live operation requires (one response per request), so no grouped sampling is needed.
- Commit: `feat(learning): bucketed moving-average baseline`

### C2-06 Monte Carlo policy gradient
- Files: `learning/rl.py`

```python
def rl_update(controller, batch, baseline, opt, beta):
    """
    Purpose:  Stage B update from one batch of on-policy episodes.
    Inputs:   batch with per-round log_probs (sum over active heads), entropy,
              rewards, complexity_band, lam index, mask; all [B, T_max]
    Notes:    G_t = sum of rewards from t to episode end (undiscounted, short episodes).
              A_t = G_t - baseline[band_t, lam]; normalise A over real rounds.
              loss = -mean(A.detach() * logp) - beta * mean(entropy).
              Clip grad norm 1.0. Then update the baseline buckets.
    """
```

- Loop: sample lam per episode from `[0.1, 0.3, 0.6, 1.0, 2.0]`, put it in `TaskRep.lam`, roll out 256 episodes with `mode="sample"`, update, repeat. Entropy beta 0.01 decaying to 0.001. Start from the supervised checkpoint, lr 1e-4.
- Test: on a tiny environment where one model is always right and cheapest, the policy converges to it.
- Commit: `feat(learning): Monte Carlo policy gradient`

### C2-07 Safety guard
- Files: `learning/safety.py`
- Build: every 250 updates, evaluate val quality at each lam. If quality falls more than 5 points below the supervised checkpoint at any lam, restore the last good checkpoint and halve the learning rate.
- Commit: `feat(learning): validation gate and rollback`

## Phase 2

### C2-08 Value head option
- Build: `ValueHead(z_H, lam)`, MSE on `G_t`, replaces the bucket baseline when `baseline: value_head`. Use only if bucketed learning curves are noisy.
- Commit: `feat(learning): optional learned value baseline`

### C2-09 Exploration for new models
- Files: `learning/exploration.py`
- Build: bonus added to a model's policy logit while its card is `warmup`: `bonus = b0 * max(0, 1 - n_uses / N)`, `b0 = 1.0`, `N = 200`. Card moves to `active` at N.
- Commit: `feat(learning): new-model exploration bonus`

### C2-10 Train with all heads
- Build: extend action branching to tools, memory, depth, verification, topology once C1-12 and C4-07 exist. Joint log-probability is the sum over heads.
- Commit: `feat(learning): RL over full strategy heads`

## Phase 3

### C2-11 Replay of live episodes
- Files: `learning/replay.py`
- Build: read `decisions`, `scores` from the `RecordStore`, rebuild episodes, fine-tune with importance weights clipped at 2.0 (logged under older policies). Offline methods such as Implicit Q-Learning are a documented next step, not required.
- Commit: `feat(learning): replay fine-tuning on logged episodes`

## Hand-offs

| You need | From | When |
|---|---|---|
| `Controller.forward_batch` | C1 | C1-06 |
| `Simulator`, `counterfactual`, episode oracle | C3 | C3-05, C3-06 |
| `CompositeScorer` | C3 | C3-07 |

| You provide | To | When |
|---|---|---|
| Supervised checkpoints, all configs, 3 seeds | evaluation | C2-04 |
| RL checkpoints for `full` and `context_free` | evaluation | C2-06 |
