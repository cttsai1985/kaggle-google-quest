import uuid

import pytest

from quest.data import TARGETS, sha256
from quest.runtime import write_json


def test_completed_run_import_is_idempotent_and_checks_artifacts(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("sqlalchemy")
    from quest.tracking import ARTIFACTS, track

    run = tmp_path / "run"
    run.mkdir()
    for name in ARTIFACTS:
        (run / name).write_text("fixture artifact\n")
    write_json(
        run / "metrics.json", {"mean_spearman": 0.5, "per_target": dict.fromkeys(TARGETS, 0.5)}
    )
    manifest = {
        "status": "completed",
        "logical_run_id": str(uuid.uuid4()),
        "config": {"name": "tracking-fixture"},
        "source": {"python_source_sha256": "fixture"},
        "artifacts": {name: sha256(run / name) for name in ARTIFACTS},
    }
    write_json(run / "manifest.json", manifest)
    previous = mlflow.get_tracking_uri()
    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    try:
        mlflow.set_tracking_uri(uri)
        client = mlflow.MlflowClient()
        with pytest.raises(ValueError, match="mlops-platform bootstrap"):
            track(run, uri)
        client.create_experiment(
            "kaggle/quest", artifact_location=(tmp_path / "artifacts").as_uri()
        )
        first = track(run, uri)
        assert first["status"] == "imported"
        imported = client.get_run(first["mlflow_run_id"])
        assert imported.data.tags["project_id"] == "quest"
        assert imported.data.tags["logical_run_id"] == manifest["logical_run_id"]
        assert imported.data.tags["source_sha256"] == "fixture"
        assert track(run, uri)["status"] == "already_imported"
        downloaded = client.download_artifacts(first["mlflow_run_id"], "model.pt")
        assert open(downloaded).read() == "fixture artifact\n"
        (run / "model.pt").write_text("changed checkpoint")
        with pytest.raises(ValueError, match="checksum"):
            track(run, uri)
        (run / "model.pt").write_text("fixture artifact\n")
        manifest["config"]["name"] = "different recipe"
        write_json(run / "manifest.json", manifest)
        with pytest.raises(ValueError, match="Conflicting"):
            track(run, uri)
        assert (
            len(client.search_runs([client.get_experiment_by_name("kaggle/quest").experiment_id]))
            == 1
        )
    finally:
        mlflow.set_tracking_uri(previous)
