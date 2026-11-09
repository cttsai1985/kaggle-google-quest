"""Verified, explicit import into an existing MLflow experiment."""

import json
import uuid
from pathlib import Path
from urllib.parse import urlparse

from quest.data import sha256
from quest.import_support import import_verified

ARTIFACTS = {
    "model.pt",
    "config.json",
    "split.csv",
    "history.json",
    "metrics.json",
    "validation.csv",
    "submission.csv",
}
OPTIONAL_ARTIFACTS = {"fold-assignments.csv", "fold-plan.json"}
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


def track(run_dir: Path, tracking_uri: str, *, resume: bool = False) -> dict:
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
        permitted = name in (ARTIFACTS | OPTIONAL_ARTIFACTS) or (
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
            "Active experiment kaggle/quest must be provisioned by an authorized MLflow operator "
            "using the client-owned namespace before import"
        )
    tags = {
        "project_id": "quest",
        "logical_run_id": logical_id,
        "manifest_sha256": digest,
        "source_sha256": manifest["source"]["python_source_sha256"],
    }
    if manifest["source"].get("git_commit"):
        tags["source_commit"] = manifest["source"]["git_commit"]
    metrics = json.loads((run_dir / "metrics.json").read_text())
    return import_verified(
        client,
        experiment_id=experiment.experiment_id,
        run_dir=run_dir,
        logical_id=logical_id,
        manifest_digest=digest,
        artifacts={**manifest["artifacts"], "manifest.json": digest},
        run_name=manifest["config"]["name"],
        tags=tags,
        params=manifest["config"],
        metrics={
            "mean_spearman": metrics["mean_spearman"],
            **{f"spearman_{key}": value for key, value in metrics["per_target"].items()},
        },
        resume=resume,
    )
