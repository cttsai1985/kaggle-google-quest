"""Explicit local training and code packaging; no automatic publication."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from quest.data import TARGETS, read_data, sha256
from quest.runtime import evaluate, predict, source_identity, train, write_json


def download(data_dir):
    executable = str(Path(sys.executable).with_name("kaggle"))
    environment = os.environ.copy()
    if not environment.get("KAGGLE_API_TOKEN"):
        result = subprocess.run(
            [executable, "auth", "print-access-token"], capture_output=True, text=True
        )
        if result.returncode or not result.stdout.strip() or "\n" in result.stdout.strip():
            raise RuntimeError("Authenticate with kaggle auth login")
        environment["KAGGLE_API_TOKEN"] = result.stdout.strip()
    data_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            executable,
            "competitions",
            "download",
            "-c",
            "google-quest-challenge",
            "-p",
            str(data_dir),
            "-q",
        ],
        env=environment,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError("Kaggle download failed; check access and competition rules")
    with zipfile.ZipFile(data_dir / "google-quest-challenge.zip") as archive:
        for name in ("train.csv", "test.csv", "sample_submission.csv"):
            matches = [member for member in archive.namelist() if Path(member).name == name]
            if len(matches) != 1:
                raise ValueError(f"Expected one {name}")
            with archive.open(matches[0]) as source, (data_dir / name).open("wb") as output:
                shutil.copyfileobj(source, output)
    receipt = {
        name: sha256(data_dir / name) for name in ("train.csv", "test.csv", "sample_submission.csv")
    }
    write_json(data_dir / "download-manifest.json", receipt)
    return receipt


def export(output):
    project = Path.cwd()
    info = project / "src/quest/build-info.json"
    write_json(info, source_identity())
    subprocess.run(["uv", "build", "--wheel"], check=True)
    assets, kernel = output / "code-dataset", output / "training"
    assets.mkdir(parents=True, exist_ok=True)
    kernel.mkdir(parents=True, exist_ok=True)
    wheel = project / "dist/quest_labeling-0.1.0-py3-none-any.whl"
    for path in (wheel, project / "LICENSE", project / "configs/deberta.json"):
        shutil.copy2(path, assets / path.name)
    write_json(assets / "source.json", {**source_identity(), "wheel_sha256": sha256(wheel)})
    write_json(
        assets / "dataset-metadata.json",
        {
            "id": "cttsai/quest-labeling-code",
            "title": "QUEST labeling code",
            "licenses": [{"name": "Apache 2.0"}],
        },
    )
    for path in (
        project / "notebooks/kaggle/train.ipynb",
        project / "notebooks/kaggle/kernel-metadata.json",
    ):
        shutil.copy2(path, kernel / path.name)
    return {"output": str(output), "wheel_sha256": sha256(wheel), "published": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "validate"):
        sub = commands.add_parser(name)
        sub.add_argument("--data-dir", type=Path, required=True)
    sub = commands.add_parser("train")
    sub.add_argument("--data-dir", type=Path, required=True)
    sub.add_argument("--run-dir", type=Path, required=True)
    sub.add_argument("--config", type=Path, default=Path("configs/deberta.json"))
    for name in ("evaluate", "predict"):
        sub = commands.add_parser(name)
        sub.add_argument("--data-dir", type=Path, required=True)
        sub.add_argument("--run-dir", type=Path, required=True)
        sub.add_argument("--device", default="cpu")
        if name == "predict":
            sub.add_argument("--output", type=Path, required=True)
    sub = commands.add_parser("export-kaggle")
    sub.add_argument("--output", type=Path, default=Path(".local/kaggle-bundle"))
    commands.add_parser("lab-token")
    sub = commands.add_parser("track")
    sub.add_argument("--run-dir", type=Path, required=True)
    sub.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    args = parser.parse_args()
    if args.command == "download":
        result = download(args.data_dir)
    elif args.command == "validate":
        frame = read_data(args.data_dir / "train.csv", labeled=True)
        test = read_data(args.data_dir / "test.csv", labeled=False)
        if set(frame.qa_id) & set(test.qa_id):
            raise ValueError("Training/test IDs overlap")
        result = {"training_rows": len(frame), "test_rows": len(test), "targets": len(TARGETS)}
    elif args.command == "train":
        result = train(args.data_dir, args.run_dir, json.loads(args.config.read_text()))
    elif args.command == "evaluate":
        result = evaluate(args.data_dir, args.run_dir, args.device)
    elif args.command == "predict":
        result = predict(args.data_dir, args.run_dir, args.output, args.device)
    elif args.command == "track":
        from quest.tracking import track

        if not args.tracking_uri:
            parser.error("Set MLFLOW_TRACKING_URI or provide --tracking-uri")
        result = track(args.run_dir, args.tracking_uri)
    elif args.command == "lab-token":
        import secrets

        directory = Path(".local")
        directory.mkdir(exist_ok=True)
        (directory / "runs").mkdir(exist_ok=True)
        token = directory / "jupyter_token"
        if token.exists():
            raise FileExistsError("Token already exists; reuse it or remove it explicitly")
        descriptor = os.open(token, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(secrets.token_hex(32) + "\n")
        result = {"token_file": str(token)}
    else:
        result = export(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
