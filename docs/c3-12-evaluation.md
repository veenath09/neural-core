# C3-12: Probe Profiler, Profile Updater, and Leave-One-Model-Out Evaluation

**Project:** A Neuro-Inspired Self-Improving Framework for Adaptive AI Agent Orchestration
**Component:** C3 — Measurement
**Task:** C3-12 — Probe Profiler and Profile Updater
**Implementation branch:** `feature/measurement`
**Implementation commit:** `00c1b3f`
**Evaluation type:** Offline, deterministic capability-aware model onboarding

---

## 1. Objective

The objective of C3-12 is to enable the orchestration system to estimate the capabilities of a model using a small set of probe observations and update its capability profile.

The implementation follows the required weighted profile update:

\[
\text{new} = \frac{w \times \text{prior} + n \times \text{observed}}{w+n}
\]

Where:

- `prior` is the existing estimated capability.
- `observed` is the capability measured using probe observations.
- `n` is the number of observations for a particular step type.
- `w = 50` is the prior weight.

The second objective is to evaluate a leave-one-model-out onboarding workflow, where one model is excluded from the initial routing pool, profiled using training observations, and introduced into the available pool without retraining the deterministic evaluation policy.

## 2. Implementation

### 2.1 ProbeProfiler

**File:** `measurement/probe_profiler.py`

The `ProbeProfiler` generates a capability vector from a model's probe observations.

Its main method is:

`ProbeProfiler.run(card, probe_set)`

The implementation:

- Requires the probe observations to belong to the training split.
- Validates the relevant probe fields and model-specific outcome columns.
- Rejects duplicate and invalid probe records.
- Calculates observed correctness for the five RouterBench task types.
- Calculates the logarithmic representation of the model's observed mean execution cost.
- Produces a six-dimensional capability vector.

**Capability vector structure:**

| Index | Capability |
|---|---|
| 0 | MMLU correctness |
| 1 | HellaSwag correctness |
| 2 | ARC-Challenge correctness |
| 3 | Winogrande correctness |
| 4 | MBPP correctness |
| 5 | Log10 mean execution cost |

The probe profiler does not modify the original model card.

### 2.2 ProfileUpdater

**File:** `measurement/profile_updater.py`

The `ProfileUpdater` updates the estimated task-specific capabilities using the weighted update equation with `w = 50`.

Its main methods are:

- `ProfileUpdater.update(card, step_type, observed, n)`
- `ProfileUpdater.update_many(card, observations)`

Each task-specific capability is updated independently.

The implementation validates input values, preserves the model's identity and metadata, and returns a new model card rather than modifying the original card in place.

The cost capability is not modified by the task-accuracy update operation.

### 2.3 Leave-One-Model-Out Experiment

**File:** `scripts/evaluate_leave_one_out.py`

The experiment evaluates model onboarding using a deterministic profile-aware routing policy.

For each fold:

1. Select one of the four RouterBench models as the held-out model.
2. Remove that model from the initial routing pool.
3. Select 20 training probes for each of the five task types.
4. Generate observed capabilities for the held-out model.
5. Construct initial task-accuracy priors from the remaining three models.
6. Apply the weighted profile update using `w = 50`.
7. Estimate the held-out model's execution cost from its training probes.
8. Add the updated model card to the routing pool.
9. Evaluate the original three-model pool and expanded four-model pool on the same test episodes.
10. Save the prediction records, summary statistics, and metadata.

The experiment uses two deterministic policies:

- **Always-cheapest:** Selects the cheapest model available in the current pool.
- **Profile-aware:** Chooses a model based on estimated task-specific capability, execution cost, and available budget.

The profile-aware policy is a heuristic baseline and is not the project's trained neural Controller.

## 3. Dataset and Evaluation Setup

**Dataset:** RouterBench
**Task types:** MMLU, HellaSwag, ARC-Challenge, Winogrande, and MBPP
**Number of models:** 4
**Training probes per fold:** 100
**Probes per task type:** 20
**Test episodes per fold:** 100
**Prior weight:** 50
**Random seed:** 0
**Evaluation split:** Test

The same selected 100 test episodes were used across all four folds.

Each fold evaluated two policies before onboarding and two policies after onboarding, producing 400 completed policy-episode evaluations per fold.

Across the four folds, this corresponds to 1,600 completed policy-episode evaluations.

Training probes and evaluation episodes were separated using the dataset split information. The experiments verified that the held-out model was absent from the initial model pool and present in the expanded pool.

### 3.1 Held-Out Models

| Fold | Held-out model |
|---|---|
| 0 | `mistralai/mistral-7b-chat` |
| 1 | `mistralai/mixtral-8x7b-chat` |
| 2 | `gpt-3.5-turbo-1106` |
| 3 | `gpt-4-1106-preview` |

## 4. Experimental Results

### 4.1 Profile-Aware Routing Results

The following table compares mean episode quality before and after onboarding.

| Fold | Model | Before | After | Change |
|---|---|---:|---:|---:|
| 0 | Mistral 7B | 0.671500 | 0.661833 | -0.009667 |
| 1 | Mixtral 8x7B | 0.665667 | 0.665167 | -0.000500 |
| 2 | GPT-3.5 Turbo | 0.661667 | 0.660833 | -0.000833 |
| 3 | GPT-4 Preview | 0.542333 | 0.553667 | +0.011333 |

**Mean quality change across the four folds: approximately +0.000083.**

The small positive average indicates an almost neutral overall result. The experiment does not establish a consistent improvement in routing quality.

### 4.2 Profile-Aware Execution Cost Results

The following table compares mean episode cost in USD.

| Fold | Model | Cost before | Cost after | Change |
|---|---|---:|---:|---:|
| 0 | Mistral 7B | 0.003854 | 0.003757 | -0.000097 |
| 1 | Mixtral 8x7B | 0.003862 | 0.003873 | +0.000011 |
| 2 | GPT-3.5 Turbo | 0.003684 | 0.003592 | -0.000092 |
| 3 | GPT-4 Preview | 0.000571 | 0.000715 | +0.000144 |

**Mean cost change across the four folds: approximately -$0.000009 per episode.**

The average difference is small, and the direction of the cost change varies between held-out models.

### 4.3 Always-Cheapest Baseline

For Fold 0, introducing Mistral 7B reduced mean episode cost from approximately $0.000427 to $0.000140, while quality decreased from 0.482000 to 0.275333.

For Folds 1, 2, and 3, the always-cheapest policy produced unchanged mean quality and cost because onboarding did not change the cheapest available model.

These results illustrate that reducing execution cost can involve a trade-off in task quality.

## 5. Interpretation

The experiments demonstrate that the offline onboarding workflow can introduce an additional model using a probe-derived capability profile.

The profile-aware heuristic produced different outcomes depending on the held-out model:

- Mistral 7B onboarding slightly reduced both mean quality and mean execution cost.
- Mixtral onboarding produced a very small quality decrease and cost increase.
- GPT-3.5 Turbo onboarding produced small decreases in both quality and cost.
- GPT-4 Preview onboarding improved mean quality while increasing mean execution cost.

The overall quality change across the four folds was close to zero.

Therefore, the results support the technical feasibility of probe-based model onboarding for this deterministic routing policy, but do not demonstrate that onboarding consistently improves routing performance.

The results are descriptive. No statistical significance or confidence-interval analysis was performed.

## 6. Automated Testing

Three dedicated C3-12 test modules were implemented:

| Test module | Tests passed |
|---|---:|
| `tests/test_measurement_probe_profiler.py` | 17 |
| `tests/test_measurement_profile_updater.py` | 48 |
| `tests/test_measurement_leave_one_out.py` | 16 |
| **Total** | **81** |

These tests cover:

- Probe field and split validation.
- Deterministic training-probe selection.
- Training/test separation checks.
- Capability-vector calculation.
- Weighted update correctness.
- Model-card metadata preservation.
- Three-model outcome table construction.
- Budget-aware heuristic routing.
- Onboarding compatibility across all four model folds.

The complete project test suite was also executed successfully.

**Test command:**

```cmd
python -m pytest -q
```

## 7. Reproducing the Experiment

Activate the project virtual environment and run the commands from the repository root.

Each evaluation must use a new evaluation ID because the script does not overwrite existing result directories.

**Fold 0 — Mistral 7B**

```cmd
python -m scripts.evaluate_leave_one_out --eval-id c3-12-repro-fold0-100 --episodes 100 --held-out-index 0 --probes-per-type 20 --seed 0
```

**Fold 1 — Mixtral 8x7B**

```cmd
python -m scripts.evaluate_leave_one_out --eval-id c3-12-repro-fold1-100 --episodes 100 --held-out-index 1 --probes-per-type 20 --seed 0
```

**Fold 2 — GPT-3.5 Turbo**

```cmd
python -m scripts.evaluate_leave_one_out --eval-id c3-12-repro-fold2-100 --episodes 100 --held-out-index 2 --probes-per-type 20 --seed 0
```

**Fold 3 — GPT-4 Preview**

```cmd
python -m scripts.evaluate_leave_one_out --eval-id c3-12-repro-fold3-100 --episodes 100 --held-out-index 3 --probes-per-type 20 --seed 0
```

Each experiment generates:

```text
results/<evaluation-id>/
    predictions.parquet
    summary.csv
    metadata.json
```

The original verified experiments were saved locally under:

- `results/c3-12-lomo-fold0-100/`
- `results/c3-12-lomo-fold1-100/`
- `results/c3-12-lomo-fold2-100/`
- `results/c3-12-lomo-fold3-100/`

These output directories are Git-ignored and are not included in the implementation commit.

## 8. Limitations

### 8.1 Trained Neural Controller Not Evaluated

The experiment uses a deterministic capability-aware routing heuristic.

It does not evaluate a trained checkpoint of the project's neural Controller.

Therefore, these results must not be presented as proof that the trained Controller can generalize to an unseen model without retraining.

### 8.2 Offline Historical Outcomes

The experiments use the existing RouterBench dataset and historical recorded model outcomes. They do not execute new requests against the model providers.

The reported execution costs are based on the dataset rather than current live provider billing.

### 8.3 Scope of Leave-One-Model-Out Validation

One model is excluded from the initial evaluation pool in each fold. Its onboarding profile is then constructed using permitted training probes.

This validates the process of adding a model to the deterministic routing pool. It is not equivalent to training a neural router on three models and testing whether that trained router generalizes to the fourth.

### 8.4 Evaluation Sample Size

Each fold uses 100 test episodes. The results should be treated as an initial offline evaluation rather than a statistically conclusive benchmark.

## 9. Conclusion

C3-12 successfully implemented and tested a probe-based model profiling and updating mechanism.

The implementation supports estimating task-specific model capabilities, applying weighted updates with a prior weight of 50, and adding a held-out model card to a deterministic capability-aware routing pool.

All 81 dedicated tests passed, and the four-fold offline evaluation completed successfully.

The experiment showed that introducing a model can change both episode quality and execution cost, but did not demonstrate consistent performance improvements across models.

**C3-12 implementation status:** Completed and pushed to GitHub.

**Outstanding research validation:** Evaluate the trained neural Controller on a genuinely held-out model using an appropriate checkpoint and controlled training setup.

---

**Commit:** `00c1b3f`
**Commit message:** `feat(measurement): probe profiler and profile updater`
**Branch:** `feature/measurement`
**Repository:** `https://github.com/veenath09/neural-core`