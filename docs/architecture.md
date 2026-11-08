# Quality prediction with shared language representations

The input consists of a title, question body and answer. Twenty-one targets describe the question; nine describe the answer. Each is a fractional annotation score in [0, 1]. A 30-way softmax would incorrectly make these dimensions compete with one another, so the model produces independent logits and uses binary cross-entropy with the supplied fractional labels.

## Two views of the same exchange

A shared pretrained encoder processes title/body and title/answer pairs. This keeps both views in the same representation space. Padding-aware pooling excludes padded positions from the mean. Concatenating the question vector, answer vector and their difference gives the prediction head information about each text and their relationship.

The maintained model uses [Microsoft DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base), pinned to revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`. That checkpoint has 12 transformer layers and hidden size 768. Its model card records the MIT license. Tokenizer files are saved with the trained model so inference needs no model-hub access.

Question and answer maximum lengths are 384 and 512 tokens respectively, retaining the original input budget. Truncation is explicit and therefore part of the experiment configuration. The head produces all 30 targets; it does not assume every target depends exclusively on one input field.

## Validation at the question level

Multiple answers can belong to one question. GroupKFold keeps each question group wholly within one partition. The maintained first fold follows the original category/question-body coding, with 4,863 training and 1,216 validation rows. It is a holdout experiment, not five-fold out-of-fold evaluation.

Mean Spearman ranks predictions within each target, then averages the 30 correlations. Ties use SciPy's average-rank convention. A constant prediction or constant target contributes zero rather than being silently excluded; all 30 target contributions are retained in the metric file. This explicit edge policy differs from the original `nanmean` treatment and is visible when relevant.

A higher validation score is useful evidence about the chosen holdout. Repeated tuning against one fold can overfit model selection. The public and private leaderboard populations are separate and must retain their own reported values.

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
