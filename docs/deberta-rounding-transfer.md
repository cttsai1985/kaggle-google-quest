# Does one frozen DeBERTa rounding policy transfer?

**Eight of ten** question-group calibration/assessment transfers improve Spearman. The mean paired delta is **+0.004489**, ranging from **−0.003345 to +0.019306**. Four of five split seeds have a positive bidirectional mean; seed 2027 is negative in both directions. This is evidence of a modest, split-sensitive effect, not a uniformly successful postprocessor.

The earlier five-fold fold-mean/pooled disagreement did not warrant abandoning the method without testing a single policy per assessment population. This follow-up does that, while retaining the original encoder and random-search rounder. It does not turn a selected checkpoint's holdout into independent model validation.

[Precommitted protocol](deberta-rounding-protocol.md) · [All target/fold evidence](deberta-rounding-transfer.json) · [Executed walkthrough](../notebooks/03-metric-aware-postprocessing.ipynb)

## Fixed calibration-transfer checks

The protocol/source were committed at `2f9499e2d86b6116afa7df0f1e0809e2270aecbe` before fitting. Use the saved 1,216-row holdout, preserving its actual IDs. Five predeclared split seeds (42, 1017, 2027, 3407, 7919) each produce two category-stratified question-group halves. Fit on one half and assess on the other, then reverse. Label support, thresholds and target selection use calibration rows only. All 30 targets remain in every score, with undefined correlations contributing zero.

Keep 1,000 random-search candidates, optimization seed 42 and the original target-selection margin −0.001. No seed or policy is chosen from these ten results. Checkpoint selection previously used this entire holdout, so all these results are conditional robustness diagnostics. Splits overlap: ten transfer deltas are not independent observations and do not yield a valid naive confidence interval.

| Split seed | Assessment half | Calibration / assessment rows | Raw | Selective rounding | Paired delta |
|---|---:|---:|---:|---:|---:|
| 42 | 0 | 607 / 609 | 0.386761 | 0.387532 | +0.000771 |
| 42 | 1 | 609 / 607 | 0.366926 | 0.374885 | +0.007959 |
| 1017 | 0 | 608 / 608 | 0.371534 | 0.377948 | +0.006414 |
| 1017 | 1 | 608 / 608 | 0.381533 | 0.387875 | +0.006342 |
| 2027 | 0 | 607 / 609 | 0.388571 | 0.387336 | -0.001235 |
| 2027 | 1 | 609 / 607 | 0.364301 | 0.360956 | -0.003345 |
| 3407 | 0 | 608 / 608 | 0.375462 | 0.380182 | +0.004720 |
| 3407 | 1 | 608 / 608 | 0.378477 | 0.378579 | +0.000102 |
| 7919 | 0 | 607 / 609 | 0.373747 | 0.393053 | +0.019306 |
| 7919 | 1 | 609 / 607 | 0.379110 | 0.382970 | +0.003859 |

The five bidirectional seed means are **+0.004365, +0.006378, −0.002290, +0.002411 and +0.011583**, respectively. The negative seed is retained, not filtered. Rare labels and approximately 600 calibration rows limit threshold/target-selection stability; this run does not isolate their causal effects or test training-seed variation.

## Mixing fold policies: a diagnostic that did not fix the result

The earlier five-fold pooled raw/selective scores are **0.377863 / 0.365200**. Normalizing each assessment partition's predictions separately to average-rank percentiles, without labels, preserves its within-fold Spearman but changes the pooled results to **0.375943 / 0.331452**. Selective rounding becomes worse, not better.

That rejects this normalization as a useful repair in this diagnostic. It does not prove which factor causes the original pooled loss: normalization changes cross-partition rank comparisons and tie structures, while populations differ. Neither normalized predictions nor a hand-picked split become the external candidate. The original and normalized all-target contributions remain in the aggregate evidence.

## A concrete candidate, held fixed before this follow-up

The external-test candidate is the canary-3 policy fitted on all 1,216 saved validation predictions, with SHA-256 `27ee88acbc793d99957cd3f7febcc2a2cbac34ea8e960687a99e8ec950ed448d`. Its full-holdout selective score **0.411136** is optimistic fit/evaluation evidence; it is not an additional independent gain.

Reloading that policy reproduces the already generated **476 × 30** test predictions exactly. The original prediction bytes and ID order are preserved. A private notebook is prepared locally: it performs fresh inference using the recorded checkpoint on competition-mounted test inputs, then applies the frozen source/policy snapshots. Its actual application cell, executed locally against retained inference outputs, matches every candidate value with **maximum difference 0**. Full hosted execution has not occurred; local application parity is not a new hosted inference or leaderboard result.

The notebook contains fitted calibration state and remains outside Git. It reuses existing code/checkpoint assets and prepares separate raw and postprocessed outputs and receipts. The candidate is staged locally; external notebook execution and scoring have not been performed. One external evaluation is proposed, with no leaderboard tuning or automatic serving promotion. No old leaderboard row or receipt is replaced.

## Reproduce

```bash
uv sync --locked --group dev --group lab
.venv/bin/python scripts/study_deberta_rounding.py \
  --data /absolute/path/train.csv \
  --producer /absolute/path/retained-quest-run \
  --previous /absolute/path/postprocessing-study \
  --output .local/new-rounding-transfer
```

The prior study directory contains the frozen policies and manifest generated by [the first protocol](postprocessing-study.md#reproduce-and-apply). Use committed, clean source and a fresh output directory. The script verifies producer artifacts, input hashes, question-group boundaries, the previous pooled scores and the fixed candidate's identity. Private assignments/policies/predictions stay outside Git.
