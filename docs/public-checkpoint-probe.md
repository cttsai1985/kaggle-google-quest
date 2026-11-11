# Reuse a tuned model, preserve its evaluation identity

The published Bibimorph RoBERTa checkpoint runs on current PyTorch/Transformers with **all 202 tensors loaded strictly**. With the author's numerical versions used only to recover fold IDs, modern GPU inference scores **0.419363**, versus the saved training log's **0.419461** (difference **−0.0000978**). This is close reconciliation, not exact historical numerical replay or a recovered original ID artifact.

Applying the repository author's original rounding method in two calibration/assessment directions raises the mean paired score **0.417749 → 0.428676 (+0.010927)**. However, **18/30 targets worsen in each direction**: the gains are concentrated, rather than uniform improvements in quality. This is a conditional diagnostic of a checkpoint already selected on this population; no new leaderboard score or serving promotion is claimed.

[Aggregate evidence](public-checkpoint-probe.json) · [Executed walkthrough](../notebooks/04-checkpoint-lineage.ipynb) · [First protocol](public-checkpoint-probe-protocol.md) · [Fold recovery protocol](public-fold-recovery-protocol.md) · [Original team attribution and recovered artifacts](original-solution-lineage.md)

## Keep the fine-tuning work in the checkpoint

Model training/tuning belongs to Bibimorph's authors. The reused public fold includes their StackExchange adaptation and pseudo-labeling work, learned weighting of 13 CLS representations and trained 30-target head. Its recorded setup specifies head/tail packing, a 500-token budget split 26/260/210 across title/question/answer, accumulation 2, warmup 200 and backbone LR 1e-5. The published optimizer uses 500× that learning rate for head parameters. Multi-sample dropout is a training feature; eval mode disables it.

The maintained adapter reimplements the published inference contract with modern APIs. It preserves source separators and zero padding and checks all state keys. On 50 actual rows, IDs/masks match the reviewed author packing functions using the current tokenizer. This verifies the packing logic; it does not assert that every historical tokenizer/runtime is bitwise identical.

Using a new backbone and an independently chosen training recipe would not preserve this work. The earlier DeBERTa experiment remains a separate baseline, not a substitute for the tuned historical team model or this public model. No backbone/head parameters are fitted here.

## The same fold name can assess different rows

All three reconstructions contain **1,216** assessment rows:

| Reconstruction | NumPy / scikit-learn | IDs shared with author-version reference | Raw mean Spearman |
|---|---|---:|---:|
| Current local libraries | 2.5.4 / 1.9.1 | 312 | 0.482525 |
| Current Kaggle libraries | 2.1.3 / 1.6.1 | 247 | 0.495579 |
| Author-version reference | 1.18.1 / 0.22.1 | 1,216 | **0.419363** |

The two native reconstructions share only **234** IDs. These higher scores cannot be read as better models: the assessment population changed and may overlap original model training. On the common 476 test rows, CPU/GPU raw predictions differ by at most **2.1e-6**. Two GPU runs with the native/reference populations produce identical raw test predictions. The checkpoint is held fixed while the assessment identity changes.

A temporary Python-3.6 CPU container uses the author's declared numerical versions solely to generate a reference assignment. Current libraries still run the model and rounder. Source input and assignment hashes are checked before inference. The restored score approaches the original log within 0.00010; the residual discrepancy remains recorded, and saved original IDs are still unavailable. Do not equate this reconciliation with proof of every historical training detail.

The held-out raw predictions and IDs are now retained privately. Secondary calibration IDs were reconstructed under the probe's numerical versions and independently checked against **every target score in both directions**, then preserved for reuse. Future producers should save these IDs at fit time rather than rely on reconstruction.

## Assess one frozen policy on another half

Within the reference population, use two category-stratified question-body halves with seed 42. Fit thresholds, label support and target selection on one half, assess on the other, then reverse. Keep the original 1,000 candidates, optimization seed 42 and gain margin −0.001. All 30 targets remain defined in both recorded comparisons.

| Direction | Calibration / assessment rows | Raw | Rounded | Delta |
|---|---:|---:|---:|---:|
| 0 | 604 / 612 | 0.404067 | 0.419976 | +0.015909 |
| 1 | 612 / 604 | 0.431432 | 0.437377 | +0.005945 |

The mean paired gain is **+0.010927**. Definition's mean target-level gain is **+0.298659**, contributing **+0.009955** of the all-30 macro gain; the other 29 targets' contributions sum to only **+0.000972**. This is arithmetic decomposition of the same predictions, not a causal ablation or permission to drop labels. Spelling changes sign across directions; answer plausibility has a large negative delta in one direction. Per-target guardrails matter for a consumer that needs answer quality rather than only the competition macro metric.

A separate full-population fit/evaluation gives **0.462506**, which is optimistic tuning evidence. Its frozen policy reloads and reproduces all 476 test outputs exactly. That checks application/serialization, not independent score improvement. Checkpoint selection previously used the reference population, so even the half-split checks remain conditional diagnostics, not fully nested acceptance tests.

## Reproduce with bounded dependencies

Download the public fold and matching tokenizer/configuration using the source links below. Keep the model bytes outside Git. The source archive contains `scripts/public_roberta_adapter.py`, `scripts/recover_public_fold.py` and the maintained `quest.postprocessing` module. The adapter supports CPU and CUDA and performs no downloads or training when loading explicit local paths.

```python
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, "scripts")
from public_roberta_adapter import load, predict
from quest.postprocessing import OptimalRounder

model, tokenizer = load(Path("best_model.pth"), Path("tokenizer-assets"), device="cpu")
# Calibration/evaluation populations must be explicit and appropriate for this model.
raw = pd.DataFrame(predict(model, tokenizer, evaluation_frame), columns=target_columns)
processed = approved_policy.transform(raw)
```

`tokenizer-assets` contains `config.json`, `vocab.json` and `merges.txt`; the study hashes its actual assets. Its 12-layer backbone config and 30-target head are validated against the state dict. The full probe used source-verified data/assignments and the protocol's separate calibration populations; the example is an application sketch, not a claim that arbitrary rows are unseen by a borrowed model.

For this reference-ID recovery only, mount the original training CSV read-only and an empty output directory:

```bash
docker run --rm --cpus 1 --memory 768m \
  -v "$QUEST_DATA_DIR:/data:ro" -v "$PWD/.local/reference-folds:/out" \
  -v "$PWD/scripts/recover_public_fold.py:/recover.py:ro" \
  python:3.6.15-slim-buster@sha256:e10aa83604948c6d8d9f72a9a20193d84bb2dbe550b725eb5208387117fde065 \
  sh -c 'python -m pip install --quiet numpy==1.18.1 scipy==1.4.1 scikit-learn==0.22.1 joblib==0.13.2 && python /recover.py'
```

Compare the assignment hash with the aggregate receipt and carry the verified file forward. Do not use this compatibility container for model training/inference; the project runtime remains Python 3.12/3.13 with its locked modern dependencies. Public release assets exclude raw rows, assignments, borrowed/private checkpoint weights, fitted policies and private notebook payloads.

Sources: [public fine-tuned fold collection](https://www.kaggle.com/datasets/kashnitsky/google-qa-quest-labeling-bibimorph-model-3-roberta), [author model/packing implementation](https://github.com/oleg-yaroshevskiy/quest_qa_labeling/tree/yorko/step5_model3_roberta_code), [recorded author dependencies](https://github.com/oleg-yaroshevskiy/quest_qa_labeling/blob/yorko/requirements_full.txt).
