# Quality prediction with shared language representations

The input consists of a title, question body and answer. Twenty-one targets describe the question; nine describe the answer. Each is a fractional annotation score in [0, 1]. A 30-way softmax would incorrectly make these dimensions compete with one another, so the model produces independent logits and uses binary cross-entropy with the supplied fractional labels.

## Two views of the same exchange

A shared pretrained encoder processes title/body and title/answer pairs. This keeps both views in the same representation space. Padding-aware pooling excludes padded positions from the mean. Concatenating the question vector, answer vector and their difference gives the prediction head information about each text and their relationship.

The maintained model uses [Microsoft DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base), pinned to revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`. That checkpoint has 12 transformer layers and hidden size 768. Its model card records the MIT license. Tokenizer files are saved with the trained model so inference needs no model-hub access.

Question and answer maximum lengths are 384 and 512 tokens respectively, retaining the original input budget. Truncation is explicit and therefore part of the experiment configuration. The head produces all 30 targets; it does not assume every target depends exclusively on one input field.

## Validation at the question level

Multiple answers can belong to one question. The current default uses five-way `StratifiedGroupKFold`, stratifying the declared category domain while keeping each exact question body wholly within one partition. Seed 42 and protocol `category_stratified_group_v1` are fixed in advance; there is no seed search.

The analyzer compares this plan with ordinary question-group folds. It records row/group coverage, category distance, target histograms and missingness, constant targets, and the number of question groups carrying nonmodal labels. Category balance does not guarantee multi-target balance, so the outcomes are checked rather than assumed.

In the measured dataset, maximum category total-variation distance falls from 0.03880 to 0.00120. Every stratified validation fold contains variation in all 30 targets, including the rare `question_type_spelling` label. Five-fold TF-IDF/Ridge CV checks empirical dispersion under this plan; each fold fits its own vocabulary and IDF on training text only. Its OOF results do not constitute transformer cross-validation.

The earlier full transformer benchmark retains its original single question-group holdout, with 4,863 training and 1,216 validation rows. Its recipe remains in `configs/benchmark.json`, and checkpoint evaluation always uses saved indices. The new default recipe and the frozen benchmark are intentionally distinguishable.

### Metric coverage

Mean column-wise Spearman evaluates within-target ranks. The maintained primary metric averages all 30 contributions, assigning zero to undefined constant-target or constant-prediction correlations. A second score averages only targets whose held-out labels vary; a constant prediction still contributes zero. Both scores and the number of defined targets are reported.

The benchmark holdout has one constant label column, `question_type_spelling`: its primary score uses 30 contributions, while the defined-target score uses 29. All five new stratified folds have 30 defined targets, so both summaries agree there. The original `nanmean` implementation could silently exclude undefined correlations; metric coverage now accompanies the result.

### Frozen assignments across runtimes

A fixed seed does not promise identical group assignments across scikit-learn or NumPy versions. The CV run saves `fold-assignments.csv` with its checksum, input-file checksum, protocol, seed and source library versions. Training can consume this file as the authority instead of rerunning the splitter. It rejects changed inputs, duplicate/missing IDs, invalid fold values, protocol mismatches and question-group leakage.

Export includes competition-derived fold IDs only when explicitly requested; code-only packaging remains the default. A retrieved checkpoint needs neither a splitter nor the full training dataset for test inference.

## Training and artifacts

The recipe initializes a prediction head over the pinned encoder, warms the head with the encoder frozen, and then fine-tunes the model with AdamW. Warmup and cosine decay govern the fine-tuning learning rate; gradient accumulation increases the effective batch size. CUDA uses mixed precision for training. Checkpoint selection maximizes held-out mean Spearman, with early stopping after two non-improving fine-tuning epochs.

A completed run retains configuration, train/validation roles, per-epoch history, per-target metrics, validation probabilities, test probabilities, the best model and tokenizer. Its manifest records input hashes, package versions, source identity, model revision, elapsed time and artifact checksums. Failed runs retain a failed manifest instead of silently resembling completed runs.

## Changes from the 2020 pipeline

| Original | Maintained implementation | Reason |
|---|---|---|
| TensorFlow/Keras transformer graph | Current PyTorch and Transformers APIs | Supported model and optimization interfaces |
| BERT/distilRoBERTa presets | Pinned DeBERTa-v3-base recipe | A modern model candidate evaluated against the historical score |
| Axis-sensitive convolution/pooling head | Padding-aware pooling and a compact prediction head | Explicit sequence semantics and fewer implementation-specific assumptions |
| Exponential learning-rate growth plus callbacks | AdamW, warmup, cosine decay and gradient clipping | One inspectable optimization schedule |
| Implicit local model directories | Complete checkpoint and tokenizer bundle | Offline inference and artifact portability |
| Unnumbered single-fold execution inside a CV loop | Explicit fold configuration and saved split IDs | Clear evaluation scope |

These are deliberate model and implementation changes. Matching or exceeding a historical score would demonstrate a modern replacement's performance; it would not establish identical TensorFlow/PyTorch numerical behavior.

The unchanged historical code and notebooks remain in `legacy/`. Its archived shell commands are retained as source history, not the maintained execution interface.

Seeds control initialization and sampling, but GPU kernels, mixed precision and library builds can change floating-point results. Saved assignments, checkpoints and prediction artifacts anchor verification; bitwise equality across platforms is not claimed.
