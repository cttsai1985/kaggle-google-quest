# QUEST evaluation results

[Hard-case diagnostics and next experiments](error-analysis.md) · [Acceptance audit](requirements-audit.md)

## Metric-aware post-processing

The original solution's random-search rounding and selective target application are restored in maintained code. A predeclared nested TF-IDF/Ridge assessment improves mean outer-fold Spearman **0.326727 → 0.345104** across five folds; pooled OOF improves **0.326114 → 0.331345**. All five outer-fold deltas are positive. These are fixed-recipe baseline results, not a new transformer or leaderboard score.

The saved DeBERTa diagnostic is mixed: selective rounding improves grouped calibration-crossfit fold mean **0.375006 → 0.382324**, but pooled Spearman falls **0.377863 → 0.365200**. Checkpoint selection used this entire holdout. Same-label fit/evaluation gives **0.411136**, an explicitly optimistic diagnostic. Raw inference remains the default and no new predictions were submitted.

The archived notebook's **0.386 → 0.415** also reuses labels for threshold fitting, column selection and evaluation; it does not independently reproduce the historical leaderboard submission. [Protocol, all conditions, acceptance decision and reproduction](postprocessing-study.md) · [Aggregate receipt](postprocessing-study.json) · [Executed walkthrough](../notebooks/03-metric-aware-postprocessing.ipynb).

## A fixed grouped CV plan

The dataset has 6,079 training rows, 3,583 unique question bodies and 30 targets. The current protocol is `category_stratified_group_v1`: five `StratifiedGroupKFold` partitions, category-domain strata, exact question-body groups and seed 42. This plan was declared before model fitting; no seed search was performed.

The analyzer compares it with unstratified question-group folds. Maximum category total-variation distance decreases from **0.0387966** to **0.0012044**; maximum target-histogram distance decreases from **0.0438235** to **0.0361049**. Every validation fold retains all five category domains and variation in all 30 targets. No question group crosses a train/validation boundary, and every row is held out once.

`question_type_spelling` has nonmodal labels in only eight question groups. The new plan covers those labels across all five validation folds. The analyzer reports support rather than assuming stratification can solve arbitrary rarity; fewer informative groups than folds can make full variation coverage impossible.

## Five-fold TF-IDF/Ridge baseline

Vocabulary and IDF are fitted separately on each fold's training title/body and title/answer views. The recipe fixes 12,000 vocabulary features, word 1–2 grams, minimum document frequency 3, Ridge alpha 10, LSQR tolerance 0.003 and at most 100 iterations. Outputs are clipped to [0, 1]. There is no transformer fine-tuning or learned-embedding reuse across folds.

| Fold | Validation rows | Mean Spearman | Defined targets |
|---|---:|---:|---:|
| 0 | 1,217 | 0.327172 | 30 |
| 1 | 1,216 | 0.331527 | 30 |
| 2 | 1,216 | 0.326905 | 30 |
| 3 | 1,215 | 0.325258 | 30 |
| 4 | 1,215 | 0.322771 | 30 |

Pooled OOF Spearman is **0.3261137129**. The fold mean is **0.3267266720** and descriptive sample SD is **0.0032046681**. Fold SD is not a confidence interval. The recorded run took **28.087 seconds**, with native-library threads limited to six. It measures stability for this baseline and plan; it does not establish transformer CV or future-data stability.

A repeat produced byte-identical fold assignment and OOF prediction CSVs. The clean source revision is `5e0c651e019ebc4d495195b3cff64d390d88f1b2`. The assignment SHA-256 is `68d0efdf4e3cd73f559a1767f2c842eb70c74039f45194c3f8944f0ca6eaf0d7`; the OOF SHA-256 is `a46cc152672fe1b9f7e28337b625d8d50d4e6f82e3080e9c2fc536c16625ffd8`.

The [aggregate CV receipt](cv-validation.json) retains the complete diagnostics, source/environment and artifact fingerprints. The saved assignment file is authoritative when reusing folds across scikit-learn/NumPy versions. Input hashes, exact ID membership, protocol and question-group boundaries are checked before fitting.

## Historical leaderboard benchmark

The authenticated Kaggle account history contains a submission named `submission.csv` on 10 February 2020 at 15:32:29 UTC, with public Spearman **0.43155** and private Spearman **0.40232**. The final submission that day scored public **0.42654** and private **0.40321**.

The history establishes leaderboard values. Its generic file name does not identify the exact training recipe or prove that submission came from this source revision. These leaderboard populations are separate from the local CV data and the modern model's holdout.

## Full transformer experiment

The predeclared DeBERTa experiment completed one frozen-head epoch and five full fine-tuning epochs on a Tesla T4. It used the earlier first question-group holdout: 4,863 training and 1,216 validation rows. Its exact configuration remains in `configs/benchmark.json`; the current default now uses category-stratified groups.

The best checkpoint, selected at fine-tuning epoch 4, achieved **0.3778628686 mean Spearman**. All 30 saved holdout targets vary, so its all-30 zero-fallback and defined-target summaries agree. The completed train/reload/test-inference function took **3,249.530 seconds**. All 476 downloadable test IDs and their order match the official sample.

An independent calculation joined saved validation predictions to labels by `qa_id` and correlated average ranks; it reproduced **0.37786286862973434**. Every input hash and output artifact checksum matched, and the frozen benchmark configuration matched the hosted configuration byte for byte.

A separate full CPU checkpoint reload evaluated all 1,216 saved holdout rows with the current locked libraries. It scored **0.3778488370**, an absolute difference of **0.0000140316** from the recorded CUDA mixed-precision result. This measured difference is retained rather than asserting bitwise cross-runtime equality.

### Runtime drift and authoritative IDs

Recreating the legacy split locally with newer libraries produces a different 1,216-row holdout: only **252 IDs overlap** the saved hosted holdout. The recreated local split also has a constant spelling target, while the actual hosted split does not. Confusing these partitions would misstate both model performance and target coverage.

The new CV plan is therefore reusable as a verified assignment file rather than a promise about seed behavior. A private CPU-only Kaggle check using scikit-learn **1.6.1** and NumPy **2.1.3** reconstructed the exact file created locally with scikit-learn **1.9.1** and NumPy **2.5.4**, including its byte hash. The first hosted transformer run’s configuration and checkpoint indices remain unchanged.

### Deployment and artifact checks

A separate offline Kaggle notebook successfully loaded the saved model/tokenizer and generated test predictions with internet disabled. The completed run was imported into the existing shared `kaggle/quest` MLflow experiment; repeating the import reused the same logical run. Downloading the 709 MiB checkpoint through the artifact proxy reproduced its original byte hash.

These are measured engineering and model results. They do not establish five-fold transformer performance, original-source numerical reproduction, or improvement over the historical leaderboard submission. DeBERTa-v3 uses different pretrained assets from the historical BERT/DistilRoBERTa pipeline. This is a retrospective modern baseline; late evaluation does not establish original-weight reproduction or historical-period competition eligibility.

The single late notebook-scoring request completed as submission `58139363`: public **0.35189**, private **0.32983**. It falls below the historical account result; no further tuning or submission search was performed. Stable CV and verified engineering, rather than matching a favorable historical score, define this release’s scope. The [validation receipt](validation.json) retains the completed checks and their limits.

## Leaderboard percentile context

A score becomes easier to interpret when compared with other teams. The official **public** leaderboard export provides this reference population; the private leaderboard is not included in that download.

| Submission | Public score | Score-equivalent position | Top share of reference teams | Strictly outperforms |
|---|---:|---:|---:|---:|
| Historical account submission, 2020-02-10 15:32 UTC | 0.43155 | 82 / 1,572 | 5.22% | 94.78% |
| Historical final submission, 2020-02-10 | 0.42654 | 104 / 1,572 | 6.62% | 93.45% |
| Modern DeBERTa late notebook submission | 0.35189 | 1193 / 1,572 | 75.89% | 24.17% |

Snapshot: **2026-11-09T00:02:27Z**, **1,572 unique scored teams** in the [official public leaderboard](https://www.kaggle.com/competitions/google-quest-challenge/leaderboard?tab=public). Smaller top-share percentages and larger outperformed percentages are better.

Position is `1 + teams with a strictly better displayed score`; equal scores share the starting position and the table retains the full tie interval. Top share is that starting position divided by the reference-team count. The outperformed percentile counts only strictly worse scores, so tied teams are excluded. Comparisons use the exported decimal scores without a tolerance.

These are **score-equivalent placements**, not awarded final ranks for late submissions. The dated snapshot is the denominator; it may differ from competition-era participant counts. Public/private leaderboard populations and internal OOF/holdout populations are separate: no private or CV percentile is inferred. The [aggregate receipt](leaderboard-context.json) records source hashes, tie counts, formulas and scope.

## DeBERTa rounding transfer follow-up

Five predeclared two-half question-group split seeds yield ten calibration-to-assessment transfers. Eight improve; mean paired Spearman delta is **+0.004489**, with one seed negative in both directions. The selected checkpoint's holdout remains a diagnostic population, not independent transformer validation. Partition-wise rank normalization does not repair the earlier pooled loss. A single canary-3 policy is fixed as the external-test candidate, and its 476×30 application outputs match exactly after reload. Hosted execution/scoring await explicit authorization; no new leaderboard result exists. [All results and limitations](deberta-rounding-transfer.md).
