# QUEST evaluation results

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

The saved assignment file is authoritative when reusing folds across scikit-learn/NumPy versions. Input hashes, exact ID membership, protocol and question-group boundaries are checked before fitting.

## Historical leaderboard benchmark

The authenticated Kaggle account history contains a submission named `submission.csv` on 10 February 2020 at 15:32:29 UTC, with public Spearman **0.43155** and private Spearman **0.40232**. The final submission that day scored public **0.42654** and private **0.40321**.

The history establishes leaderboard values. Its generic file name does not identify the exact training recipe or prove that submission came from this source revision. These leaderboard populations are separate from the local CV data and the modern model's holdout.

## Full transformer experiment

The predeclared DeBERTa run is training on a private Tesla T4 notebook. It uses the earlier first question-group holdout, with 4,863 training and 1,216 validation rows. Its exact configuration remains in `configs/benchmark.json`; the current default recipe now uses category-stratified groups.

This earlier holdout has a constant `question_type_spelling` label. Its all-30 metric assigns zero to that undefined correlation; the defined-target counterpart averages the remaining 29. The result and target coverage will be recorded when the full recipe completes. No reproduced or improved leaderboard result is claimed while that measurement is pending.
