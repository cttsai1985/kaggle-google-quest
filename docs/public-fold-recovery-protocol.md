# Recover the public model's assessment identity

The first probe succeeded in loading/inference, but native fold reconstruction
produced two different 1,216-row populations: only 234 IDs overlap between the
current CPU and Kaggle libraries. Their scores differ from each other and from
the author's recorded 0.4194612505. Raw predictions on the common 476 test rows
agree within 2.1e-6; the assessment-population mismatch is separately observable.
Retain both first-probe results as diagnostics, not independent CV improvements.

Recover one fixed reference partition using the author's declared numerical
versions: NumPy 1.18.1, scikit-learn 0.22.1, SciPy 1.4.1 and joblib 0.13.2, in the
Python 3.6 family. This temporary CPU container generates indices only from the
read-only original CSV. The first Python-3.8 attempt failed during old joblib
import; it produced no model or score. The successful reference uses
`python:3.6.15-slim-buster`, image digest
`sha256:e10aa83604948c6d8d9f72a9a20193d84bb2dbe550b725eb5208387117fde065`.
Model loading, inference and post-processing remain on current libraries.

Use only the recovered fold-0 assignment. Its SHA-256 is
`1cf428a6fd023cac8212f61948eda81c44e4e04e9c15e03a7c1c1019097dd3c0`;
it contains 6,079 source rows and 1,216 assessment rows. Verify input and assignment
bytes, group boundaries and order before assessment. Keep the same public
checkpoint, tokenization/head, two-way calibration split (seed 42), 1,000-candidate
rounder, optimization seed 42 and -0.001 margin. Save the new held-out raw
predictions with IDs so the population need not be reconstructed again.

Compare raw Spearman with the author's recorded best score, reporting both
fixed-30 and defined-target means. A close score reconciliation supports this
version-based reconstruction but does not replace missing original ID artifacts
or establish every original training detail. Do not search other library versions,
folds, checkpoints or seeds to improve the score. Retain any disagreement.
No additional leaderboard submission or serving promotion is part of this probe.
