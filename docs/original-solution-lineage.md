# Recover the team solution before replacing its model

The historical leaderboard models were trained and tuned by a teammate. The
repository author's contribution is metric-aware post-processing. The later
DeBERTa experiment and checkpoint adapters are retrospective engineering and
validation work. These roles should not be collapsed into a claim that this
repository author trained the original leaderboard models.

A deeper account audit recovered the team's exact submission-to-notebook links,
private notebook source and saved outputs, plus four older fine-tuned checkpoints.
It did **not** recover the best team's RoBERTa weights or numerically replay the
best result. The earlier statement that a generic `submission.csv` prevented
identifying the source was incomplete: structured Kaggle history retains the
notebook URL and script-version ID.

[Aggregate lineage evidence](original-solution-lineage.json) ·
[Modern post-processing evaluation](deberta-rounding-transfer.md) ·
[Public-checkpoint probe protocol](public-checkpoint-probe-protocol.md)

## The historical score has a team and a version

| Team submission | Public / private Spearman | Source identified by authenticated Kaggle history |
|---|---|---|
| 14440999, 2020-02-10 | 0.43155 / 0.40232 | `guozhiyu0914/post-pro-roberta-5fold`, script version 28452096 |
| 14442348, 2020-02-10 | 0.42654 / 0.40321 | `guozhiyu0914/bert-roberta`, script version 28458995 |

Both records identify **Five Experts** and the teammate's submitting account.
The author separately clarified the training contribution; submitting a file or
owning a dataset is not by itself proof of who trained every component.
The corresponding private notebook sources and cached 476-row outputs were
retrieved through the account's normal access. Source/output fingerprints and
schema/ID checks are retained, while private code and prediction rows stay out
of Git. The API's current v5 source is retained separately: a byte-level mapping
from that source/output to the historical script-version ID is not independently
verified, and no numerical reproduction is claimed from the cached file alone.

## There are several different model pipelines

| Pipeline | Confirmed input/head contract | Recovery status |
|---|---|---|
| Older local experiments | BERT, DistilBERT and two DistilRoBERTa variants; two views and a difference vector; 2304×30 dense head | Four actual fine-tuned HDF5 checkpoints recovered and inspected |
| Retrieved best-team RoBERTa source | 384-token question and answer branches; concatenate their pooled representations; eight model-loading iterations despite the notebook's “5fold” title; specialized rare-label handling | Source and cached output recovered; referenced `roberta-s` weights unresolved |
| Retrieved final-team source | RoBERTa components plus six BERT-loading iterations | Source and cached output recovered; checkpoint inputs unresolved |
| Retrospective DeBERTa baseline | Shared encoder, padding-aware means and question/answer/difference head; separate controlled experiment | Full checkpoint and raw/rounded score receipts available |
| Borrowed Bibimorph RoBERTa probe | Single combined 500-token input; head/tail budgets; learned weighting of 13 CLS states | Public fine-tuned fold downloaded; all 202 state tensors strictly loaded in current PyTorch/Transformers |

The older private dataset is `cttsai/google-quest-challenge-weights`, dated
7 February 2020. Its four `tf_model_fine-tuned.h5` files really contain trained
30-target heads. They are not evidence that the later best-team RoBERTa ensemble
has been recovered. Its HDF stores retain aggregate diagnostics, without the
row-level calibration labels/predictions needed to replay threshold fitting.

The retrieved teammate source references `roberta-s`, `ro-tokenizer-l` and
`bert5fold`. Current metadata returns unresolved dataset references. A direct
lookup of the **guessed** owner/slug `guozhiyu0914/roberta-s` returns HTTP 403;
this does not establish the real dataset owner or prove the weights were deleted.
Exact checkpoint recovery remains an access/identity gap, not a reason to invent
a replacement model and label it historical reproduction.

## The post-processing was target-specific

The retrieved team notebook contains more than a generic rounding call. It fits
OptimalRounder on 27 targets, retains separate paths for three rare targets,
uses domain information for spelling, and contains distribution discretization
and a specialist not-really-a-question model. Calibration predictions are made
on the training frame by one component, rather than supplied as a verified full
OOF matrix. Preserve that historical scope; its fit metric is not independent
validation. The leaderboard result is a separate external evaluation.

These are source observations, not permission to redistribute the private
notebook or proof that every retrieved source line produced the exact historical
submission bytes. The maintained rounder captures one important component; it is
not yet the complete original team ensemble and all special-label policies.

## “Newer backbone” does not reproduce fine-tuning

The modern baseline already includes a frozen-head stage, separate head/backbone
learning rates, warmup and grouped evaluation. It does not reproduce every
original team or public winner recipe. The public checkpoint's saved settings and
published source additionally establish StackExchange adaptation, pseudo-labels,
head/tail token allocation, learned layer weighting, a 500× head/backbone learning
rate ratio, accumulation and multi-sample dropout during training.

Those components were learned by the original public model authors. Reusing their
checkpoint preserves them without claiming new training work. The adapter uses
current framework APIs, loads every weight strictly, and checks token IDs/masks
against reviewed author packing functions on 50 actual rows. It intentionally
preserves the original input contract, including zero padding. Changing these
while “cleaning up” would change the model being evaluated.

This list does not establish which trick caused a score difference. That needs
controlled ablations. The immediate goal is a faithful model consumer with clear
lineage and a calibrated prediction contract, followed by an honest test of the
repository author's post-processing.

Sources: [published inference/model implementation](https://github.com/oleg-yaroshevskiy/quest_qa_labeling/tree/yorko/step5_model3_roberta_code),
[public fine-tuned checkpoint collection](https://www.kaggle.com/datasets/kashnitsky/google-qa-quest-labeling-bibimorph-model-3-roberta),
and authenticated account records summarized in the linked aggregate receipt.
