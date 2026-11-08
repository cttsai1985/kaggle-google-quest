"""Verified, explicit import into an existing MLflow experiment."""

import json
import uuid
from pathlib import Path
from urllib.parse import urlparse

from quest.data import sha256
from quest.runtime import write_json

ARTIFACTS = {
    "model.pt",
    "config.json",
    "split.csv",
    "history.json",
    "metrics.json",
    "validation.csv",
    "submission.csv",
}
TOKENIZER_FILES = {
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "spm.model",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "tokenizer.model",
}


def track(run_dir: Path, tracking_uri: str) -> dict:
    import mlflow
    from mlflow import MlflowClient

    if urlparse(tracking_uri).username:
        raise ValueError("Use runtime authentication rather than credentials in the URI")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("project_id", "quest") != "quest":
        raise ValueError("This client imports QUEST runs only")
    if manifest["status"] != "completed":
        raise ValueError("Only completed runs can be imported")
    logical_id = manifest["logical_run_id"]
    uuid.UUID(logical_id)
    for name, digest in manifest["artifacts"].items():
        path = Path(name)
        permitted = name in ARTIFACTS or (
            path.parent == Path("tokenizer") and path.name in TOKENIZER_FILES
        )
        artifact = run_dir / path
        if not permitted or artifact.is_symlink() or not artifact.is_file():
            raise ValueError(f"Unexpected artifact: {name}")
        if sha256(artifact) != digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    if not ARTIFACTS <= set(manifest["artifacts"]):
        raise ValueError("Required artifacts are missing")
    digest = sha256(run_dir / "manifest.json")
    mlflow.set_tracking_uri(tracking_uri)
    name = "kaggle/quest"
    client = MlflowClient()
    experiment = client.get_experiment_by_name(name)
    if experiment is None or experiment.lifecycle_stage != "active":
        raise ValueError(
            "Provision the active kaggle/quest experiment with the mlops-platform bootstrap "
            "before importing: https://github.com/cttsai1985/mlops-platform"
        )
    matches = client.search_runs(
        [experiment.experiment_id], f"tags.logical_run_id = '{logical_id}'"
    )
    if matches:
        previous = matches[0]
        if (
            previous.data.tags.get("manifest_sha256") != digest
            or previous.info.status != "FINISHED"
        ):
            raise ValueError("Conflicting or incomplete import for this logical run")
        return {"status": "already_imported", "mlflow_run_id": previous.info.run_id}
    tags = {
        "project_id": "quest",
        "logical_run_id": logical_id,
        "manifest_sha256": digest,
        "source_sha256": manifest["source"]["python_source_sha256"],
    }
    if manifest["source"].get("git_commit"):
        tags["source_commit"] = manifest["source"]["git_commit"]
    with mlflow.start_run(
        experiment_id=experiment.experiment_id, run_name=manifest["config"]["name"], tags=tags
    ) as active:
        mlflow.log_params(manifest["config"])
        metrics = json.loads((run_dir / "metrics.json").read_text())
        mlflow.log_metric("mean_spearman", metrics["mean_spearman"])
        mlflow.log_metrics(
            {f"spearman_{key}": value for key, value in metrics["per_target"].items()}
        )
        for filename in [*manifest["artifacts"], "manifest.json"]:
            parent = Path(filename).parent
            mlflow.log_artifact(
                str(run_dir / filename), artifact_path=None if parent == Path(".") else str(parent)
            )
        result = {"status": "imported", "mlflow_run_id": active.info.run_id}
    write_json(run_dir / "tracking.json", result)
    return result
