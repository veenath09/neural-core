# Decision log

Design decisions with the evidence and alternatives considered.

## 2026-10-21 Threshold selection rule
Claim: a cost-aware controller should be judged at a cost no higher than a simple baseline, not at its most expensive setting.
Evidence: with "max val quality" the depth-pool context_free run chose tau 0.99 at $0.0117/episode (2x random, 3.5x other runs) for +0.5 points; the whole quality-cost curves of all configs overlap.
Alternatives: max val quality (favours expensive settings); quality minus lambda x cost (needs a lambda we have no basis for); area under the quality-cost curve (harder to read).
Decision: best val quality among settings whose val cost is <= the random-option baseline's, ties within 0.002 to the cheaper; fall back to the cheapest setting if none qualifies. Report quality at matched cost caps and the full curve beside it. Fixed before any test number.
Supervisor discussion: pending.

## 2026-10-21 Role of the reasoning-depth pool
Claim: the depth pool (19 model x depth options) tests whether the controller learns when to think deeper; it cannot test the context claim.
Evidence: 80 val / 179 test questions; correlated episodes are built only from all-correct questions, so episode-type effects cannot be measured. Synthetic data cannot fix evaluation because val and test must stay real.
Alternatives: collect more real questions for the depth pool (API spend; recommended hard bands 3-4 and GSM8K, about $25-35 per 1,000 questions); drop the depth pool.
Decision: the RouterBench pool is the main experiment for the context claim. The depth pool is a second experiment reporting difficulty prediction (AUC), quality at fixed cost, and the depth chosen per difficulty band.
Supervisor discussion: pending.

## 2026-10-21 Synthetic training questions for the depth pool
Claim: borrowing outcomes from the most similar collected question gives training labels that carry real per-question signal.
Evidence: leave-one-out on 633 real train questions: P(real right | borrowed right) 0.952 vs P(real right | borrowed wrong) 0.774 (+0.18) when matching on the 11 RouterBench model scores; +0.03 matching on a mean-score bin; -0.005 for a random same-task donor. Outcome distribution reproduced (all-correct 79.6% vs 78.4%).
Alternatives: LLM-generated questions (outcomes would still need real API calls); per-option logistic imputation (loses the joint pattern across options); more real collection (API spend).
Decision: hot-deck imputation (data.depth_synth), training split only, rows labelled `synthetic` with `donor_id`; model cards from real data only; every reported number on real val/test.
Supervisor discussion: pending.

## 2026-10-21 Loss weights
Claim: the spec weights (model 1.0, halt 0.5, complexity 0.2) apply.
Evidence: the code summed the three terms 1/1/1, an open item in the build guide to fix before the analysis freeze.
Decision: 1.0/0.5/0.2 in train/rollout.py (LOSS_WEIGHTS). Earlier 1/1/1 runs in runs/ are superseded.
Supervisor discussion: not needed (matches the spec).

## 2026-10-22 Reward from an independent audit verdict
Claim: when the controller acts on a verifier (escalate when flagged), the same verdict must not be its reward.
Evidence: the decision verdict was selected by escalation (flagged rounds re-rolled), read high (mean episode reward > 1)
and the online learner escalated to re-roll it (Kimi run: 3.2x cost, lower true quality).
Alternatives: true quality in simulation (unavailable live); a KL anchor to the start (blocked genuine adaptation).
Decision: rewards use an independent audit draw of each step's final answer, de-biased with the known flip rate;
decisions keep the usual verdict. Live exact-match verification is deterministic and needs no audit.
Supervisor discussion: pending.

## 2026-10-22 Escalation means a strictly pricier card
Claim: "escalate" must mean more expensive, not later in the pool list.
Evidence: models joining at run time are appended whatever their price, so a cheap joiner became an escalation target.
Decision: escalation targets are cards with a strictly higher cost (Episode(model_costs), rollout, session); cost-ordered
pools are unchanged. Found by the other session's review (finding 4).

## 2026-10-22 A simulator with hidden episode difficulty for the context claim
Claim: the context claim needs workflows where history carries information the current prompt does not.
Evidence: the standard simulator changes outcomes from history on 0.4% of steps; full and context_free tie there.
Alternatives: real agent traces (preferred long term, C3); stronger retry/propagation rules (harder to interpret).
Decision: opt-in Episode(hard_p): a hard episode (probability hard_p, fixed per episode) fails the cheaper half of the
models on every step. Default off keeps C3's simulator. Primary hypothesis H2 is tested on it (analysis plan).
Supervisor discussion: pending; to propose to C3 as a benchmark extension.

## 2026-10-22 Live pilot through the depth collector, under an estimate-based cap
Claim: the live pilot should use the 2026 options we measured, with their thinking parameters.
Evidence: C4's ExecutionEngine has no adapters for these options; data.depth_collect.call_arm is tested on all of them.
Decision: scripts/online_live.py uses call_arm and exact-match verification; a pre-call estimate (3x mean cost) and a
post-call charge enforce the cap; --dry-run makes no call. Not run live: needs the user's go-ahead and dollar cap.

## 2026-10-22 Analysis frozen before seeds 1-2
Decision: docs/analysis_plan.md and eval/final_test.py fixed and tagged `analysis-freeze` before any seed 1-2 or test
number. H2 (context, hidden-difficulty simulator, paired test contrast over 3 seeds) is the single primary hypothesis.
