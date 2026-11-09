# QUEST: the MLOps consumer contract

The solution owns the modeling question, validation method and acceptance criteria. It consumes shared execution, tracking and storage services. Its scientific claims do not become stronger merely because a run appears in MLflow.

## Why the evaluation partition travels with the model

The hosted-to-local comparison exposed a portability problem that a successful model load would miss. Both regenerated and saved holdouts had 1,216 rows, yet only **252 IDs overlapped**. The declared seed and grouping procedure did not preserve the same evaluation population across library versions.

The engineering decision is to carry evaluation identity as an artifact: input hash, actual row assignments, protocol and group-boundary checks. A checkpoint is evaluated on its saved indices; a supplied fold plan is authoritative only after its IDs and input fingerprint validate. Otherwise a plausible score could describe a different cohort and different target coverage while appearing to reproduce the original run.

This [observed cross-runtime comparison](results.md) explains the requirement more directly than a list of pinned packages. Environment reproducibility and evaluation identity solve related but distinct problems. The inference handoff also preserves model/tokenizer identities and official test IDs; the model's transferred bytes were independently checked. Same-run import recovery is additionally covered by controlled fixtures, rather than presented as an observed outage of this model.

## From scientific evidence to an executable job

| Boundary | This solution supplies | What it expects from shared services |
|---|---|---|
| Inputs | Authorized data, schema/ID checks and exact input hashes | Read-only input access and explicit identity on execution |
| Experiment | Fixed configuration, folds, seeds, preprocessing and budget | An immutable environment/image reference and recorded task state |
| Evaluation | OOF/holdout analyzer, retained weak folds, variance and ablation evidence | Durable results and logs for both favorable and unfavorable experiments |
| Delivery | Verified checkpoint and ordered batch outputs | Artifact transfer with checksummed readback; explicit failure/recovery |
| Troubleshooting | A portable local run bundle and project-specific validation errors | Stage/flow correlation, owner-visible status and safe retry semantics |

This repository owns project ID `quest`, tracking experiment `kaggle/quest` and artifact namespace `projects/quest`. These existing identities remain unchanged. An authorized operator provisions the experiment through MLflow using this client-declared configuration; the importer requires it to exist and never chooses a fallback experiment. The operator checks the intended artifact route before provisioning. Shared services do not need a catalog entry or source-code change for this solution. Run identities and source/input/model fingerprints remain valid across local and offline Kaggle execution.

## Failure behavior matters to research

A transient tracker outage must not require retraining an expensive completed experiment. `track --resume` recovers the same logical/remote run only after the operator confirms the earlier importer has stopped. A conflicting manifest or duplicate logical identity fails closed. Artifact delivery is marked finished only after byte-hash readback. [Project-specific commands and recovery details](reproduction.md).

A poor model remains an experiment worth recording. A delivery gate may reject a candidate without hiding its metrics. Acceptance thresholds must be declared for the intended use and evaluated on the right population; a convenient competition holdout threshold is not production approval.

## Online serving handoff

The solution owns its inference interface and model acceptance criteria. A serving release must package a supported MLflow Model with an input/output signature, compatible dependencies and all fitted preprocessing. For QUEST, that includes tokenizer identity, text truncation, target order and the declared model/ensemble recipe. A raw checkpoint or an artifact uploaded to MLflow is not by itself an MLflow serving package; packaging and serving validation remain implementation work in this repository.

The handoff supplies a verified immutable model version or image digest, prediction checks, resource requirements and declared evaluation gates. The shared platform owns deployment, access control, health/latency/error monitoring, traffic switching and rollback. A registry alias may select a release candidate, but a running release must record the resolved version; changing an alias does not establish that serving processes have reloaded it. The solution links to the [shared infrastructure requirements](https://github.com/cttsai1985/mlops-platform/blob/main/docs/requirements.md); the infrastructure does not import this repository or maintain its project identity.

## Evidence and limits

The [modeling study](modeling-study.md) tests component contributions and descriptive variability. The [shared batch case study](https://github.com/cttsai1985/mlops-platform/blob/main/docs/batch-operations.md) demonstrates an Iceberg saved-model workflow with real inputs, task logs, delivery gating, failure and recovery. It does not imply every historical experiment or every solution ran under that scheduler.

Current CI and recovery tests exercise the client contracts. The [capability assessment](https://github.com/cttsai1985/mlops-platform/blob/main/docs/mlops-assessment.md) distinguishes these controls from live service monitoring, automatic retraining, model promotion and disaster recovery. This repository makes no claim to operate those production capabilities.
