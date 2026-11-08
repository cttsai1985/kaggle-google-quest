# Train, inspect and deploy a QUEST experiment

## Local environment and data

Install Python 3.12 and uv, then use `uv sync --locked --group dev --group lab --group data`. This creates a project-local `.venv`; it does not reuse the host's global scientific packages. The source supports Python 3.12–3.13, including the inspected current Kaggle image. CPU dependencies are selected through a dedicated PyTorch CPU index; hosted CUDA dependencies come from Kaggle's image and are recorded separately.

Set `QUEST_DATA_DIR` to an absolute directory outside Git. Authenticate with `.venv/bin/kaggle auth login`, accept the competition rules, and run:

```bash
.venv/bin/quest download --data-dir "$QUEST_DATA_DIR"
.venv/bin/quest validate --data-dir "$QUEST_DATA_DIR"
```

The download command captures the Kaggle access token privately and obtains the official archive. It writes only the three expected payloads and their hash receipt. Do not commit data or authentication files.

## Training and independent evaluation

```bash
.venv/bin/quest train --data-dir "$QUEST_DATA_DIR" \
  --config configs/deberta.json --run-dir .local/my-run
.venv/bin/quest evaluate --data-dir "$QUEST_DATA_DIR" --run-dir .local/my-run
.venv/bin/quest predict --data-dir "$QUEST_DATA_DIR" --run-dir .local/my-run \
  --output .local/my-run/reloaded-submission.csv
```

Use a new run directory for each experiment. CPU execution is supported for tests, exploration and inference; full transformer training is intended for a GPU. The recipe uses eight CPU threads and gradient accumulation. Changing the model, token lengths, fold or learning schedule changes the experiment and must remain in the resolved configuration.

Evaluation reconstructs the model from its saved encoder configuration, loads its state, verifies the exact training-file hash and uses saved holdout indices. Tokenization loads locally from the run bundle. Prediction preserves test ID and target order and produces an output checksum receipt.

## Docker JupyterLab and dev container

```bash
.venv/bin/quest lab-token
docker compose -f docker/compose.yaml up --build -d lab
docker compose -f docker/compose.yaml stop lab
```

Lab binds to http://127.0.0.1:8892 and requires the token in `.local/jupyter_token`. Data is mounted read-only at `/data`, source at `/workspace`, and run outputs at `/outputs`. The container uses a non-root UID 1000 account with eight CPU cores, up to 24 GiB RAM and 256 processes. Adjust host UID/write permissions if required. The dev container uses the same Lab image and data-directory variable.

An existing SeaweedFS client mount can be added read-only:

```bash
export SEAWEED_MOUNT=/absolute/path/to/existing/client-mount
docker compose -f docker/compose.yaml -f docker/compose.seaweed.yaml up -d lab
```

This exposes `/shared` and deploys no storage service. Do not mount a storage server's internal volume.

## Kaggle training

```bash
.venv/bin/quest export-kaggle
.venv/bin/kaggle datasets create -p .local/kaggle-bundle/code-dataset
.venv/bin/kaggle kernels push -p .local/kaggle-bundle/training \
  --accelerator NvidiaTeslaT4 --timeout 28800
.venv/bin/kaggle kernels status cttsai/quest-deberta-training
.venv/bin/kaggle kernels logs cttsai/quest-deberta-training
.venv/bin/kaggle kernels output cttsai/quest-deberta-training -p .local/kaggle-output
```

Export prepares code-only assets locally. Dataset creation and kernel push are explicit account writes; metadata defaults to private. Use `datasets version` for later asset versions rather than attempting to create an existing slug. Adjust the `cttsai` Kaggle owner when using another account.

The training notebook obtains the pinned public pretrained model with internet access and saves a complete trained run. Its attached code wheel is verified against the source receipt before installation. The inspected hosted runtime uses Python 3.13 and a CUDA PyTorch build; its package set is recorded in the run instead of being described as identical to the local lock.

## Offline scoring

QUEST is a notebook-only competition. Prepare the completed model/tokenizer/manifest as a private trained-run dataset, retaining its artifact hashes. Attach it and the same code wheel to `notebooks/kaggle/inference.ipynb`, with the accompanying inference metadata. Internet must be disabled for scoring.

The inference notebook checks the completed manifest, verifies each artifact, loads model configuration/state and tokenizer locally, and writes `/kaggle/working/submission.csv` against the competition-mounted test rows. A scoring request names that notebook's exact version and output file:

```bash
.venv/bin/kaggle competitions submit google-quest-challenge \
  --kernel cttsai/quest-deberta-inference --version VERSION \
  --file submission.csv --message "Recorded modern QUEST benchmark"
```

Late-scoring availability and submission acceptance are determined by Kaggle. A local or hosted public-test file alone does not establish a competition score. Retain the returned submission ID, notebook version, artifact identities and completed score.

## Existing MLOps services

The run directory is portable and does not require a private server to survive training. Shared MLflow, SeaweedFS and any orchestration services remain outside this repository; their deployment, upgrades, credentials and backups are infrastructure concerns. A Kaggle worker need not reach the private network. Retrieve its completed run, verify it locally and import deliberately into the chosen tracker.

The optional MLflow client validates a completed run and imports it explicitly:

```bash
uv sync --locked --group dev --group lab --group data --group tracking
export MLFLOW_TRACKING_URI=http://your-existing-tracker:5000
.venv/bin/quest track --run-dir .local/my-run
```

Use an address reachable from the runtime. Import verifies artifact hashes and rejects conflicting or incomplete repeats of the same logical run ID. Tracker availability does not affect saved local training artifacts. The client is tested with an isolated SQLite MLflow backend, including artifact retrieval, duplicate import and conflicting-run rejection; the existing shared server remains a separate integration check.

An existing SeaweedFS client mount is a filesystem route; MLflow's artifact proxy is a separate route. Avoid distributing storage administration credentials to notebooks. Prefect orchestration and W&B mirroring need a configured external endpoint and their own end-to-end evidence before being described as active integrations.
