# DeBERTa rounding transfer: fixed follow-up

Continue the original rounder, not a new threshold-search method. Fix 1,000
candidates, random-search seed 42 and selective gain margin −0.001. The candidate
for a possible single external evaluation is the already generated full-holdout
policy from canary 3; do not choose among seeds or methods using the new results.
Its policy SHA-256 is
`27ee88acbc793d99957cd3f7febcc2a2cbac34ea8e960687a99e8ec950ed448d`.

Use retained DeBERTa predictions on the actual 1,216-row question-group holdout.
Verify producer input, predictions and split hashes and align labels by qa_id.
At each predeclared split seed 42, 1017, 2027, 3407 and 7919, partition the holdout
into two category-stratified question-group halves. Fit thresholds, reference label
support and target selection on one half; assess the single frozen policy on the
other, then reverse the direction. Report all ten paired raw/rounded scores, all
30 target contributions, target coverage, both directions' average per seed and
win counts. Primary diagnostic is the average paired assessment delta; overlapping
partitions mean these ten deltas are not independent and do not define a CI.

This is a robustness diagnostic of postprocessing conditional on a selected
checkpoint. Earlier checkpoint selection used all these labels, so this does not
create independent transformer validation or support automatic serving promotion.
No encoder refit or GPU is needed. Do not replace old receipts or leaderboard rows.

For the earlier five-fold diagnostic, additionally normalize each assessment
partition's outputs to its own average-rank percentiles, separately for raw and
rounded predictions. This uses no labels and leaves within-partition Spearman
unchanged; show original and normalized pooled scores. It diagnoses sensitivity
to mixing scales/ties across different policies, not causal proof or independent
performance. This transductive normalization is not the submission candidate.

Reload the fixed canary-3 policy, reproduce its full-holdout diagnostic and separate
476-row test predictions, verify order and source-file hashes, and prepare a private
CPU notebook that applies that same frozen policy to the retained producer's raw
test predictions. Keep full-holdout fit/evaluation labeled optimistic. Local and
hosted parity are application checks, not leaderboard scores. Any external scoring
is a separate single, explicitly authorized submission with no leaderboard tuning.
