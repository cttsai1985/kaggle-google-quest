# Public QUEST checkpoint reuse: fixed probe

Reuse the published Bibimorph RoBERTa fold-0 checkpoint without any encoder/head
training. Preserve the author's 13-layer learned CLS weighting, five-sample head
in eval mode, 500-token title/question/answer packing (26/260/210 budget), head/tail
truncation, separators and original zero padding. Verify every state-dict key and
compare packing with reviewed author functions before GPU execution. Use current
PyTorch/Transformers, not an old training stack. Do not redistribute borrowed or
team-private weights; record their hashes and sources.

Public dataset: kashnitsky/google-qa-quest-labeling-bibimorph-model-3-roberta.
Reference: oleg-yaroshevskiy/quest_qa_labeling, yorko,
step5_model3_roberta_code/{model,dataset}.py. Its recorded configuration includes
StackExchange adaptation, pseudo-labels, five folds, backbone LR 1e-5, head LR 500x
backbone LR, accumulation 2 and warmup 200. This is one borrowed tuned fold, not
our training contribution or the winner's full ensemble.

Reconstruct the declared fold-0 population using native GroupKFold(5), grouping
question_title in original training CSV order. No saved ID assignments are
available in this public collection. Record the assignment hash and library
versions and compare the resulting score with the author's training log, but do
not call this independently verified original OOF membership. A score match alone
cannot prove exact ID identity. This limitation remains even if the score improves.

Split that reconstructed population into two category-stratified question-body
halves (seed 42). Fit the existing OptimalRounder only on one half and assess on
the other; reverse direction and publish both deltas. Keep all 30 targets, with
undefined correlations contributing zero, and also report the original defined-
target mean for comparison with legacy logs. Use the existing fixed 1,000-candidate
search, seed 42 and gain margin -0.001. Do not tune the checkpoint, split, search
budget or selection margin after seeing results. These are conditional diagnostic
results because original checkpoint training membership is unverified, not a new
independent model acceptance score.

Fit one separate candidate policy on the entire reconstructed population and
apply it to fresh 476-row test predictions, preserving raw outputs and ID order.
Label its fit/evaluation score optimistic. This validates public-checkpoint reuse
and application of our original method; no new leaderboard scoring, serving
promotion, winner-result claim or original-team numerical replay is in this probe.
