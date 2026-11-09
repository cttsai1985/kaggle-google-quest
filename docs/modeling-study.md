# What does each text field contribute?

## Finding

Adding the question body and then the answer improves the lightweight text baseline on every one of three predeclared grouped split plans. The effects are much larger than the observed variation between these plans. This identifies a useful input representation; it does not establish transformer architecture quality.

| Input information | Mean OOF Spearman across 3 split plans | Split-plan range | Sample SD across plans |
|---|---:|---:|---:|
| Title only | 0.220134 | 0.219740–0.220702 | 0.000504 |
| Title + body | 0.294286 | 0.293442–0.295146 | 0.000852 |
| Title + body + answer | 0.325638 | 0.325071–0.326114 | 0.000527 |

| Matched split seed | Add body: Δ mean Spearman | Add answer: Δ mean Spearman |
|---|---:|---:|
| 42 | +0.074532 | +0.031842 |
| 43 | +0.074444 | +0.030582 |
| 44 | +0.073481 | +0.031629 |

## Which targets benefit?

The aggregate improvement hides a tradeoff. Adding the answer raises the **nine answer-target** mean strongly, but slightly reduces the **21 question-target** mean in all three plans. The overall metric weights all 30 targets equally.

| Input information | Mean across question targets | Mean across answer targets |
|---|---:|---:|
| Title only | 0.256508 | 0.135262 |
| Title + body | 0.341852 | 0.183301 |
| Title + body + answer | 0.340272 | 0.291489 |

The answer-addition effect on question-target means ranges from **−0.002154 to −0.001116**, while answer-target means gain **+0.104545 to +0.111165**. Because both slots share a refitted vocabulary, this experiment does not isolate the mechanism behind the question-target decrease. A separate target-specific representation would be a justified next hypothesis; no such improvement is claimed here.

## Hypothesis and controlled comparison

The labels describe both question and answer quality. A title may identify topic and wording, but cannot describe all detail in a body or whether an answer is useful. The hypothesis is that each additional field contributes ranking information beyond the title.

All variants retain **two text slots**: `(title, title)`, `(title + body, title)`, then `(title + body, title + answer)`. The first transition changes only the body content of slot one; the second changes only the answer content of slot two. Both slots share a vocabulary fitted only on each fold’s training views. Vocabulary size is capped at 12,000; concatenation therefore has at most 24,000 columns. Adding content can change the learned vocabulary and IDF; those are consequences of the input-component change, not separately tuned parameters.

The Ridge alpha (10), LSQR tolerance (0.003), iteration cap (100), word 1–2 grams, minimum document frequency (3), 24,000-character cap per view and clipped outputs are unchanged. No learned transformer embeddings enter this comparison.

## What varies—and what does not

All 6,079 labeled rows are held out exactly once in each of five category-stratified question-group folds. The seed-42 assignment matches the retained frozen assignment file. Seeds **43 and 44** are additional predeclared split plans; each variant uses exactly the same assignments within a plan. No repeated question body crosses the fit/validation boundary. All three plans and all variants are retained.

Ridge with this solver has no meaningful stochastic model seed. Repeating its fit under new “model seeds” would manufacture evidence. The reported variability instead measures sensitivity to **three grouped split plans**. It is descriptive, not a confidence interval, training randomness estimate or guarantee about future domains. Fold SD would measure another quantity and is not used as a standard error.

## Verification and budget

The completed study fitted **45 fold models** using two CPU threads in **118.5 seconds**. All nine retained OOF files were independently reloaded, their input/assignment/output hashes and ID coverage checked, and mean Spearman recomputed. Maximum metric delta after CSV round-trip: **0**. Raw question text and OOF rows remain outside Git. An initial run of the same recipe was retained privately; the final run binds these results to the published script hash after code formatting and lint cleanup. No outcomes or seeds were selected or dropped.

```bash
uv sync --locked --group dev --group lab
.venv/bin/python scripts/ablation_study.py \
  --data "$QUEST_DATA_DIR/train.csv" \
  --frozen-plan .local/cv-run/fold-assignments.csv \
  --output .local/my-ablation --threads 2
```

Use a new output directory. The frozen plan is a data-derived private artifact: obtain it by running the documented [CV workflow](reproduction.md) on the authorized competition data, rather than publishing its row IDs. [The aggregate study receipt](modeling-study.json) records actual configurations, assignment/model-output fingerprints, per-target and per-fold metrics, runtime and source identity. [The executed companion notebook](../notebooks/02-controlled-ablations.ipynb) visualizes the matched effects.

## Decision and limits

Retain both body and answer views for the lightweight baseline. A transformer still needs its own controlled architecture/training ablations; comparing it directly with Ridge would change too many factors to explain a causal component effect. These CV results use training rows and cannot be mapped to leaderboard percentiles. The [leaderboard context](leaderboard-context.json) compares only actually scored public submissions.
