# Optimize the metric, then test what generalizes

The original QUEST solution treats **post-processing as a modeling decision**. With Spearman, a strictly increasing rescaling cannot improve ranks, but discretization can change ties. The maintained code now restores the original per-target random-search rounder and selective application, with explicit calibration boundaries and reloadable state.

A strict nested TF-IDF/Ridge assessment improves the predeclared five-fold mean from **0.326727 to 0.345104 (+0.018378)**, with a positive selective-rounding delta in all five outer folds. Pooled OOF improves more modestly, **0.326114 → 0.331345**. This supports the method on this baseline and split plan; it does not reproduce the historical leaderboard model or establish an improved transformer submission.

[Executed walkthrough](../notebooks/03-metric-aware-postprocessing.ipynb) · [Aggregate evidence](postprocessing-study.json) · [Protocol committed before fitting](postprocessing-protocol.md)

## What the original contribution does

`legacy/nlp_utils/Utils/OptimalRounder.py` searches 1,000 sorted uniform threshold candidates per target with seed 42 and maps predictions onto observed label values. The original analysis notebook applies targets whose calibration gain exceeds **−0.001**, excluding undefined correlations. It displays **0.386 → 0.415 (+0.029)** after fitting thresholds and selecting columns on the same validation labels. That is the archived notebook's optimistic fit/evaluation result, not an independent gain or proof of which code produced the account's historical **0.43155 / 0.40232** public/private scores.

The current implementation preserves this search and clipped K-cut/K-value mapping. It obtains reference label values from calibration rows only. It also handles constant predictions as zero correlations instead of allowing a NaN initialization to freeze search. Those are explicit modern safety changes; this is method preservation rather than exact legacy numerical replay. All maintained comparisons retain the fixed 30-target denominator, including undefined correlations as zero.

## Separate fitting the model from fitting the postprocessor

Five category-stratified question-group outer folds assess all 6,079 labeled rows once. Within each outer-training set, three grouped inner folds produce OOF predictions. Each inner model fits its own vocabulary, IDF and Ridge coefficients on its training rows only. Only the inner OOF predictions and outer-training labels fit thresholds, label support and column selection. A refit on all outer-training rows supplies predictions for the untouched outer-assessment rows.

The recipe, seeds 42/43, 1,000 candidates and −0.001 selection margin were committed before this run. There are 20 CPU model fits and no transformer retraining. The primary outcome is the mean outer-fold all-30-target Spearman. Fold SD is descriptive, not a confidence interval; one split plan does not measure seed sensitivity.

| Condition | Fold mean Spearman | Descriptive fold SD | Pooled OOF Spearman |
|---|---:|---:|---:|
| Raw | 0.326727 | 0.003205 | 0.326114 |
| Strict monotone control | 0.326727 | 0.003205 | 0.326114 |
| Round all targets | 0.337643 | 0.007947 | 0.318571 |
| Selective rounding | 0.345104 | 0.007390 | 0.331345 |

All-target rounding improves the fold mean while **reducing pooled OOF**. Different fold policies assign different tie groups and label scales. Concatenating their predictions creates cross-fold rank comparisons absent from within-fold assessment. Reporting both prevents a favorable summary from hiding this effect. The raw baseline differs from the earlier six-thread receipt by approximately 1e-7 in fold mean under this three-thread run; the earlier evidence remains unchanged.

## The modern transformer: encouraging diagnostic, no promotion

The retained DeBERTa checkpoint's saved 1,216-row holdout reproduces raw **0.377863**. Fitting and evaluating on those same labels raises selective rounding to **0.411136**. This is an explicitly optimistic diagnostic, not a new independent validation result.

Splitting the saved holdout into five grouped calibration/assessment partitions gives:

| Condition | Fold mean Spearman | Descriptive fold SD | Pooled OOF Spearman |
|---|---:|---:|---:|
| Raw | 0.375006 | 0.015941 | 0.377863 |
| Strict monotone control | 0.375006 | 0.015941 | 0.377863 |
| Round all targets | 0.380791 | 0.021455 | 0.367247 |
| Selective rounding | 0.382324 | 0.022477 | 0.365200 |

The fold mean increases **0.375006 → 0.382324**, while pooled Spearman **falls 0.377863 → 0.365200**. Furthermore, the checkpoint was originally selected using this entire holdout: cross-fitting only its postprocessor does not remove that earlier selection bias. These diagnostics cannot justify a production policy or a leaderboard improvement. The fold raw mean also differs from the whole-holdout metric because Spearman is not additive over partitions.

A single saved policy was reloaded and applied to a separate copy of all **476** test predictions, preserving IDs/order and the original file hash. That verifies the application path only; these outputs have not been scored, submitted or promoted. The current benchmark, default raw inference and serving adapter remain unchanged.

## The engineering decision

Treat the fitted postprocessor as a versioned part of the prediction contract: retain target order, calibration population, source/input fingerprints and policy checksum; apply the same frozen policy at inference. Keep policy fitting away from acceptance labels. A JSON geometry check verifies valid schema and cut points, while external checksums establish artifact identity; it is not a complete tamper-proof evidence validator.

Before promoting a transformer policy, fit it using independent training OOF predictions and assess the complete frozen model-plus-policy on a separate question-group population. If several base models are ensembled, declare whether calibration occurs before or after ensembling, then test that exact order. The present evidence leaves that acceptance gate open. More threshold tuning against this same holdout would not close it.

## Reproduce and apply

With authorized data and the retained completed transformer run outside Git:

```bash
uv sync --locked --group dev --group lab
.venv/bin/python scripts/study_postprocessing.py \
  --data /absolute/path/train.csv \
  --transformer-run /absolute/path/quest-run \
  --output .local/postprocessing-new-run
```

The command requires committed, clean source and a new output directory. It retains private row-level OOF predictions and JSON policies alongside an aggregate manifest. The executed public walkthrough uses only the published aggregate evidence and a synthetic tie example; it can run without competition data or a GPU.

```python
import pandas as pd
from quest.postprocessing import postprocess_predictions

# An explicit policy trained on calibration data, never on inference labels.
raw = pd.read_csv("raw-predictions.csv")
processed = postprocess_predictions(raw, "approved-policy.json")
processed.to_csv("postprocessed-predictions.csv", index=False, mode="x")
```

The helper validates the declared 30 targets and unique IDs, preserves row order and leaves the source frame intact. Approval and independent assessment of a particular policy remain the caller's responsibility. No additional project CLI is needed.
