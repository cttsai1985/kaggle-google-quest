# Metric-aware post-processing: predeclared validation

The original solution's defining post-processing is per-target random-search
thresholding (`legacy/nlp_utils/Utils/OptimalRounder.py`) plus selective application
in `legacy/notebook/02_result_analysis.ipynb`. Preserve that idea rather than
substituting an unrelated calibrator or attributing the solution only to its encoder.

Spearman measures ranks and ties. A strictly increasing mapping preserves the
score; discretization changes tie groups and can change it. Four fixed conditions:
raw predictions; their square as a rank-preserving control; learned discretization
on all targets; and selective discretization with the original calibration gain
criterion `gain > -0.001`, excluding undefined correlations. Keep all 30 target
contributions and count undefined correlations as zero in maintained comparisons.

The maintained rounder retains 1,000 uniform random candidates, seed 42, the legacy
K-cut/K-value clipped mapping and calibration-only column selection. It handles
constant predictions explicitly instead of letting a NaN initialization lock the
search. This is preservation of the method, not identical legacy numerical replay.

Primary validation: **mean outer-fold all-30-target Spearman**. Also show pooled
OOF, every outer fold and target, because different fold-specific discretizers can
change cross-fold rank/tie comparability. Use the current fixed TF-IDF/Ridge recipe,
five category-stratified question-group outer folds (seed 42), and three grouped
inner folds (seed 43). Every inner vocabulary/IDF/model fits inner training only;
inner OOF predictions on outer-training rows fit the rounder. Its label support,
thresholds and selection never see outer-assessment labels. An outer model fitted
on all outer-training rows supplies the raw outer-assessment predictions. No model
family, split seed, candidate budget or margin is tuned after seeing results.
This is a retrospective fixed-recipe experiment, not an untouched competition test.

Separately audit the saved DeBERTa checkpoint's 1,216-row holdout, preserving its
actual indices and metric. Show same-label fit/evaluation as an explicitly
optimistic diagnostic and grouped cross-fitting of the postprocessor as a second
diagnostic. The base checkpoint was selected using that holdout, so repartitioning
its saved predictions does not create fully nested Transformer CV or an unbiased
new model score. Do not conflate either with the strict nested Ridge assessment.

Preserve the archived notebook's rounded **0.386 → 0.415 (+0.029)** with its scope:
threshold fitting, column selection and evaluation reused validation labels. It
is not a newly reproduced independent gain or proof of the historical submission's
exact source. No GPU training, new Kaggle scoring/submission, raw benchmark
replacement or automatic policy promotion is part of this study.

Keep fitted thresholds, row-level predictions and IDs private. Publish aggregate
comparisons, source/input/split hashes and an executed walkthrough. Apply an explicit
saved policy to a separate copy of predictions; preserve the original prediction
file and sample order. No fitted policy becomes a deployment default without its
own acceptance evidence.
