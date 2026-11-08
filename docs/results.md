# QUEST experiment results

## Historical benchmark

The authenticated Kaggle account history contains 89 submission records. The selected historical benchmark is submission `submission.csv` on 10 February 2020 at 15:32:29 UTC, with public Spearman **0.43155** and private Spearman **0.40232**. The final submission on that date scored public **0.42654** and private **0.40321**.

The history establishes leaderboard values. A generic file name does not identify the exact training recipe or prove those submissions came from this source revision. The maintained model targets comparable competition scoring while recording its own recipe and artifacts.

## Modern experiment

The complete dataset has been validated: 6,079 training rows, 476 downloadable test rows and 30 targets. The first question-group fold contains 4,863 training rows and 1,216 validation rows.

The full-data DeBERTa GPU experiment has been submitted as a private Kaggle notebook. Its outcome is pending and no reproduced or improved benchmark is claimed yet. The tiny-transformer software test is separate from model-performance evidence.

## Verified execution checks

- Four local tests pass, including a complete tiny-transformer training/evaluation/prediction cycle.
- The official data passes input and ID contracts.
- A private Kaggle runtime probe completed on Tesla T4 with Python 3.13, PyTorch 2.11.0+cu130, Transformers 5.20.2, NumPy 2.1.3, pandas 2.3.3 and scikit-learn 1.6.1.
- Model-hub access reached the pinned public DeBERTa checkpoint revision.

The GPU training result, independent checkpoint evaluation, complete hosted inference and competition score must be recorded before benchmark reproduction is marked complete.
