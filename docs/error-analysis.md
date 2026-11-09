# Where the retained transformer struggles

The retained DeBERTa model's errors make **answer relevance and plausibility** more useful follow-up targets than a raw list of the lowest per-label correlations. Its weakest spelling correlation has only one nonmodal validation label and a strong tie constraint. The analysis also finds no simple “all hard examples are long” pattern. These observations suggest a targeted input experiment and better rare-label support before another architecture change.

The full retained holdout remains **1,216 question/answer pairs across 716 question groups and all 30 targets**. Mean Spearman independently recomputes to **0.3778628686**, exactly matching the saved result. No row, label or category is removed; the selected checkpoint, predictions and headline metric are unchanged. This is a retrospective audit of that selected-checkpoint holdout, separate from the Ridge input ablation and from leaderboard scores. [Aggregate evidence](error-analysis.json) · [Full result context](results.md).

## An error diagnostic that respects the ranking task

For each target, assign average ranks to tied true and predicted values, then take their absolute rank difference divided by `n - 1`. A row's diagnostic is the equally weighted mean over all 30 targets. The predeclared cohort is the highest **10%**, rounded up: **122 rows**. Exact diagnostic ties use retained prediction-file order.

This **absolute normalized rank gap is not an additive contribution to Spearman** and is not an error probability. It helps locate disagreements without changing the objective. The selected rows contain **13.97% of total rank-gap mass**; that is a share of this diagnostic, not a share of lost competition score. Labels are needed to form the cohort, so it cannot be used as an unseen-case routing rule.

| Diagnostic | Hardest 122 rows | Remaining 1,094 rows |
|---|---:|---:|
| Mean gap across all targets | 0.324264 | 0.222679 |
| Mean gap across 21 question targets | 0.309265 | 0.218943 |
| Mean gap across 9 answer targets | 0.359260 | 0.231394 |

The answer-family mean Spearman is **0.336175**, versus **0.395729** for the question family. These families contain different labels and support patterns, so the difference alone is not a causal test of a shared encoder or head.

## Low correlation can mean different things

| Target | Retained Spearman | Nonmodal truth rows | Ceiling under the observed rank-tie pattern |
|---|---:|---:|---:|
| `question_type_spelling` | 0.039434 | 1 | 0.049650 |
| `question_not_really_a_question` | 0.065888 | 15 | 0.191184 |
| `answer_plausible` | 0.106684 | 284 | 0.739974 |
| `answer_relevance` | 0.109846 | 231 | 0.682996 |
| `answer_helpful` | 0.209084 | 481 | 0.877234 |

The final column sorts the two observed average-rank vectors in the same order, giving the maximum correlation attainable by **reordering that prediction-rank multiset** against those true ranks. It holds observed ties fixed. It is an analytical diagnostic, not a replacement score or an intrinsic ceiling for every future model.

For spelling, one truth value occurs in 1,215 of 1,216 rows. A nearly distinct prediction ordering cannot reach correlation one against that tied truth, even with the unusual observation ordered correctly. The low raw correlation therefore does not by itself justify prioritizing spelling-model changes. One informative observation also provides little basis for a generalization claim. In contrast, relevance and plausibility retain hundreds of nonmodal labels and much more headroom under the observed tie patterns. The next investigation should inspect the supported answer-quality tasks rather than optimize the sparsest label in isolation.

## Inspect domain and length before choosing a remedy

**34 of 247 Stack Overflow pairs** enter the cohort: **13.77%**, compared with the all-row cohort rate **10.03%**. Their representation is **1.37×** the validation population share. This is an unadjusted descriptive association; category mix, label support and annotation patterns have not been isolated. Every category remains in the evaluation, and the report retains support counts for all five.

Hard-cohort answer length has median **463.5 characters**, versus **567.5** for the remaining rows. Question-body medians are **442.5** versus **538**. These medians do not identify unusually long inputs as the general explanation. They also cannot exclude truncation for a smaller subset: characters are not tokenizer tokens, and medians do not describe every example. A token-level truncation audit should precede a larger context budget.

## From pattern to a controlled next experiment

**Observation:** answer relevance and plausibility have substantial ranking disagreements with meaningful label support. **Hypothesis:** the answer representation may need more of the actual question context. The current model separately encodes question and answer views and combines pooled `q`, `a` and `q - a` representations; its answer view receives the title and answer. This mechanism suggests a test, not a proven cause. [Model](../src/quest/model.py).

A bounded next comparison would add question-body context to the answer-side input while fixing the encoder, initialization/training seeds, training budget, answer-side token budget and accepted grouped assignments. Record how that reallocation truncates question and answer text. Report all 30 targets, question/answer families, supported domains and the overall metric. Do not select new hyperparameters against these 122 already-inspected rows; use a predeclared grouped evaluation that is independent of this diagnostic's tuning decisions. The [existing Ridge study](modeling-study.md) provides separate evidence about input fields, not a transformer result for this proposed change.

Rare-label work needs enough informative question groups in evaluation. The current stratified-group analyzer already reports support; it should expose inadequate support rather than manufacture confidence with repeated deterministic fits.

## Reproduce the audit

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
.venv/bin/python scripts/analyze_hard_cases.py \
  --run-dir .local/kaggle-output-v2/quest-run \
  --train "$QUEST_DATA_DIR/train.csv" \
  --output .local/quest-hard-cases.json
```

Use a new output path. The script verifies source/retained-file hashes, exact IDs, target order, split coverage and question-group separation, then recomputes the metric. Known-rank fixtures test ties, rare-label correlation ceilings and identity failures. The public receipt contains aggregates only; no question text, answers, IDs or individual predictions are published. [Analyzer](../scripts/analyze_hard_cases.py) · [Tests](../tests/test_hard_cases.py).
