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
