"""Explicit local training and code packaging; no automatic publication."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from quest.data import TARGETS, plan_from_cv, read_data, sha256
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


def export(output, owner="cttsai", cv_run=None):
    project = Path.cwd()
    info = project / "src/quest/build-info.json"
    write_json(info, source_identity())
    subprocess.run(["uv", "build", "--wheel"], check=True)
    assets = output / "code-dataset"
    assets.mkdir(parents=True, exist_ok=True)
    wheel = project / "dist/quest_labeling-0.1.0-py3-none-any.whl"
    for path in (wheel, project / "LICENSE", project / "configs/deberta.json"):
        shutil.copy2(path, assets / path.name)
    for name in ("fold-assignments.csv", "fold-plan.json"):
        if (assets / name).exists():
            (assets / name).unlink()
    if cv_run is not None:
        plan = plan_from_cv(cv_run)
        shutil.copy2(plan["path"], assets / "fold-assignments.csv")
        plan["path"] = "fold-assignments.csv"
        write_json(assets / "fold-plan.json", plan)
        config = json.loads((assets / "deberta.json").read_text())
        config["fold_plan"] = plan
        write_json(assets / "deberta.json", config)
    expected = {wheel.name, "LICENSE", "deberta.json", "source.json", "dataset-metadata.json"}
    if cv_run is not None:
        expected |= {"fold-assignments.csv", "fold-plan.json"}
    for path in assets.iterdir():
        if path.name not in expected or path.is_symlink() or not path.is_file():
            raise ValueError("Unexpected staging files; choose a fresh output directory")
    write_json(
        assets / "source.json",
        {
            **source_identity(),
            "wheel_sha256": sha256(wheel),
            "contains_data_derived_folds": cv_run is not None,
        },
    )
    write_json(
        assets / "dataset-metadata.json",
        {
            "id": f"{owner}/quest-labeling-code",
            "title": "QUEST labeling code",
            "licenses": [{"name": "Apache 2.0"}],
        },
    )
    for destination, notebook, metadata in (
        ("training", "train.ipynb", "kernel-metadata.json"),
        ("inference", "inference.ipynb", "inference-metadata.json"),
    ):
        kernel = output / destination
        kernel.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / "notebooks/kaggle" / notebook, kernel / notebook)
        values = json.loads((project / "notebooks/kaggle" / metadata).read_text())
        values["id"] = owner + "/" + values["id"].split("/", 1)[1]
        for key in ("dataset_sources", "kernel_sources"):
            values[key] = [
                owner + "/" + value.split("/", 1)[1] if value.startswith("cttsai/") else value
                for value in values[key]
            ]
        write_json(kernel / "kernel-metadata.json", values)
    return {
        "output": str(output),
        "wheel_sha256": sha256(wheel),
        "contains_data_derived_folds": cv_run is not None,
        "published": False,
    }


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
    sub.add_argument(
        "--cv-run", type=Path, help="Reuse a completed CV run’s frozen fold assignments"
    )
    for name in ("evaluate", "predict"):
        sub = commands.add_parser(name)
        sub.add_argument("--data-dir", type=Path, required=True)
        sub.add_argument("--run-dir", type=Path, required=True)
        sub.add_argument("--device", default="cpu")
        if name == "predict":
            sub.add_argument("--output", type=Path, required=True)
    sub = commands.add_parser("export-kaggle")
    sub.add_argument("--output", type=Path, default=Path(".local/kaggle-bundle"))
    sub.add_argument("--owner", default="cttsai", help="Kaggle username for staged assets")
    sub.add_argument(
        "--cv-run", type=Path, help="Explicitly stage data-derived folds in the private asset"
    )
    commands.add_parser("lab-token")
    sub = commands.add_parser("track")
    sub.add_argument("--run-dir", type=Path, required=True)
    sub.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    sub.add_argument(
        "--resume",
        action="store_true",
        help="Resume an incomplete import after its previous writer has stopped",
    )
    sub = commands.add_parser("analyze-cv")
    sub.add_argument("--data-dir", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub.add_argument("--folds", type=int, default=5)
    sub.add_argument("--seed", type=int, default=42)
    sub = commands.add_parser("cv")
    sub.add_argument("--data-dir", type=Path, required=True)
    sub.add_argument("--run-dir", type=Path, required=True)
    sub.add_argument("--config", type=Path, default=Path("configs/cv.json"))
    args = parser.parse_args()
    if args.command == "analyze-cv":
        import pandas as pd

        from quest.cv import analyze

        report = analyze(pd.read_csv(args.data_dir / "train.csv"), folds=args.folds, seed=args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
        result = {
            "output": str(args.output),
            "rows": report["rows"],
            "question_groups": report["question_groups"],
            "folds": args.folds,
        }
    elif args.command == "cv":
        from quest.cv import run_cv

        report = run_cv(
            args.data_dir / "train.csv", args.run_dir, json.loads(args.config.read_text())
        )
        result = {
            key: report[key]
            for key in (
                "status",
                "model",
                "fold_mean_30_zero_fallback",
                "fold_std_30_zero_fallback",
                "elapsed_seconds",
            )
        }
        result["pooled_oof_30_zero_fallback"] = report["pooled_oof"][
            "mean_spearman_30_zero_fallback"
        ]
        result["run_dir"] = str(args.run_dir)
    elif args.command == "download":
        result = download(args.data_dir)
    elif args.command == "validate":
        frame = read_data(args.data_dir / "train.csv", labeled=True)
        test = read_data(args.data_dir / "test.csv", labeled=False)
        if set(frame.qa_id) & set(test.qa_id):
            raise ValueError("Training/test IDs overlap")
        result = {"training_rows": len(frame), "test_rows": len(test), "targets": len(TARGETS)}
    elif args.command == "train":
        config = json.loads(args.config.read_text())
        if args.cv_run is not None:
            config["fold_plan"] = plan_from_cv(args.cv_run)
        result = train(args.data_dir, args.run_dir, config)
    elif args.command == "evaluate":
        result = evaluate(args.data_dir, args.run_dir, args.device)
    elif args.command == "predict":
        result = predict(args.data_dir, args.run_dir, args.output, args.device)
    elif args.command == "track":
        from quest.tracking import track

        if not args.tracking_uri:
            parser.error("Set MLFLOW_TRACKING_URI or provide --tracking-uri")
        result = track(args.run_dir, args.tracking_uri, resume=args.resume)
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
        result = export(args.output, args.owner, args.cv_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
