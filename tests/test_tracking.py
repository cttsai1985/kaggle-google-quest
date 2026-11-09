import json
import uuid

import pytest

from quest.data import TARGETS, sha256
from quest.runtime import write_json


@pytest.mark.parametrize("interrupt_upload", [False, True])
def test_completed_run_import_is_idempotent_and_checks_artifacts(
    tmp_path, monkeypatch, interrupt_upload
):
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
        with pytest.raises(ValueError, match="authorized MLflow operator"):
            track(run, uri)
        client.create_experiment(
            "kaggle/quest", artifact_location=(tmp_path / "artifacts").as_uri()
        )
        if interrupt_upload:
            original_upload = mlflow.MlflowClient.log_artifact
            calls = []

            def interrupted(client, run_id, local_path, artifact_path=None):
                calls.append(run_id)
                original_upload(client, run_id, local_path, artifact_path)
                if len(calls) == 2:
                    raise ConnectionError("Injected connection loss after partial upload")

            with monkeypatch.context() as fault:
                fault.setattr(mlflow.MlflowClient, "log_artifact", interrupted)
                with pytest.raises(ConnectionError, match="partial upload"):
                    track(run, uri)
            failed_id = calls[0]
            assert client.get_run(failed_id).info.status == "FAILED"
            receipt = json.loads((run / "tracking.json").read_text())
            assert receipt["mlflow_run_id"] == failed_id
            assert receipt["error_type"] == "ConnectionError"
            assert receipt["status"] == "failed"
            with pytest.raises(ValueError, match="--resume"):
                track(run, uri)
            # Stale RUNNING state also requires the explicit operator decision.
            client.update_run(failed_id, status="RUNNING")
            with pytest.raises(ValueError, match="--resume"):
                track(run, uri)
            first = track(run, uri, resume=True)
            assert first["status"] == "resumed"
            assert first["mlflow_run_id"] == failed_id
            assert client.get_run(failed_id).info.status == "FINISHED"
            assert client.get_run(failed_id).data.tags["artifact_verification"] == "verified"
        else:
            first = track(run, uri)
        assert first["status"] == ("resumed" if interrupt_upload else "imported")
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
