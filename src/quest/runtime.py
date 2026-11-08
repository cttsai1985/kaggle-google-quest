"""Full-data transformer training with portable checkpoints and run evidence."""

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from quest.data import TARGETS, TokenizedPairs, mean_spearman, read_data, sha256, split_fold
from quest.model import QuestModel


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_identity():
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    package = Path(__file__).resolve().parent
    for path in sorted(package.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())

    def git(*args):
        try:
            result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        except FileNotFoundError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    revision = git("rev-parse", "HEAD")
    build = package / "build-info.json"
    provenance = json.loads(build.read_text()) if build.exists() else {}
    if provenance.get("python_source_sha256") != digest.hexdigest():
        provenance = {}
    dirty = git("status", "--porcelain")
    return {
        "git_dirty": bool(dirty) if dirty is not None else provenance.get("git_dirty"),
        "git_commit": revision if revision is not None else provenance.get("git_commit"),
        "python_source_sha256": digest.hexdigest(),
        "legacy_commit": "ef4fb92c470e45c0a07b0ee0e474224d88d3d410",
    }


def configure(config):
    if config["threads"] < 1 or config["batch_size"] < 1 or config["epochs"] < 1:
        raise ValueError("Threads, batch size and epochs must be positive")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["threads"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def move_pair(question, answer, device):
    return (
        {key: value.to(device) for key, value in question.items()},
        {key: value.to(device) for key, value in answer.items()},
    )


@torch.inference_mode()
def probabilities(model, dataset, batch_size, device):
    model.eval()
    outputs = []
    for question, answer, _ in DataLoader(dataset, batch_size=batch_size):
        question, answer = move_pair(question, answer, device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            outputs.append(torch.sigmoid(model(question, answer)).float().cpu().numpy())
    prediction = np.concatenate(outputs)
    if not np.isfinite(prediction).all() or not ((prediction >= 0) & (prediction <= 1)).all():
        raise ValueError("Invalid output probabilities")
    return prediction


def write_predictions(path, ids, prediction):
    if prediction.shape != (len(ids), len(TARGETS)):
        raise ValueError("Prediction shape does not match IDs and targets")
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["qa_id", *TARGETS])
        writer.writerows(
            (identifier, *values) for identifier, values in zip(ids, prediction, strict=True)
        )


def train(data_dir: Path, run_dir: Path, config: dict):
    device = configure(config)
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "schema_version": 1,
        "logical_run_id": str(uuid.uuid4()),
        "status": "running",
        "source": source_identity(),
        "config": config,
        "input_sha256": {
            name: sha256(data_dir / name)
            for name in ("train.csv", "test.csv", "sample_submission.csv")
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "numpy", "pandas", "scipy", "scikit-learn")
            },
        },
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "config.json", config)
    try:
        frame = read_data(data_dir / "train.csv", labeled=True)
        test = read_data(data_dir / "test.csv", labeled=False)
        sample = read_sample(data_dir / "sample_submission.csv", test.qa_id.tolist())
        train_idx, valid_idx = split_fold(frame, config["fold"], config["folds"])
        roles = np.full(len(frame), "train", dtype=object)
        roles[valid_idx] = "validation"
        frame[["qa_id"]].assign(role=roles).to_csv(run_dir / "split.csv", index=False)
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"], revision=config.get("model_revision", "main")
        )
        tokenizer.save_pretrained(run_dir / "tokenizer")
        dataset = TokenizedPairs(frame, tokenizer, config)
        test_set = TokenizedPairs(test, tokenizer, config, labeled=False)
        training, validation = Subset(dataset, train_idx), Subset(dataset, valid_idx)
        loader = DataLoader(
            training,
            batch_size=config["batch_size"],
            shuffle=True,
            generator=torch.Generator().manual_seed(config["seed"]),
        )
        model = QuestModel(config).to(device)
        manifest["resolved_model_revision"] = getattr(model.encoder.config, "_commit_hash", None)
        history, best_score, stale = [], -math.inf, 0
        for stage, epochs in (("head", config["head_epochs"]), ("fine_tune", config["epochs"])):
            model.freeze_encoder(stage == "head")
            rate = config["head_learning_rate"] if stage == "head" else config["learning_rate"]
            optimizer = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad),
                lr=rate,
                eps=1e-8,
                weight_decay=config["weight_decay"],
            )
            accumulation = config["gradient_accumulation_steps"]
            total_steps = math.ceil(len(loader) / accumulation) * epochs
            scheduler = (
                get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=int(total_steps * config["warmup_fraction"]),
                    num_training_steps=total_steps,
                )
                if stage == "fine_tune"
                else None
            )
            mixed = device.type == "cuda" and config.get("mixed_precision", True)
            scaler = torch.amp.GradScaler("cuda", enabled=mixed)
            for epoch in range(1, epochs + 1):
                model.train()
                losses = []
                optimizer.zero_grad(set_to_none=True)
                for step, (question, answer, targets) in enumerate(loader, 1):
                    question, answer = move_pair(question, answer, device)
                    with torch.autocast(
                        device_type=device.type, dtype=torch.float16, enabled=mixed
                    ):
                        loss = nn.functional.binary_cross_entropy_with_logits(
                            model(question, answer), targets.to(device)
                        )
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite training loss")
                    scaler.scale(loss / accumulation).backward()
                    if step % accumulation == 0 or step == len(loader):
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), config["max_gradient_norm"])
                        scaler.step(optimizer)
                        scaler.update()
                        if scheduler is not None:
                            scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                    losses.append(float(loss.detach()))
                    if step % 100 == 0:
                        print(
                            json.dumps(
                                {
                                    "stage": stage,
                                    "epoch": epoch,
                                    "step": step,
                                    "steps": len(loader),
                                    "loss": np.mean(losses),
                                }
                            ),
                            flush=True,
                        )
                prediction = probabilities(model, validation, config["batch_size"], device)
                metric = mean_spearman(frame[list(TARGETS)].to_numpy()[valid_idx], prediction)
                row = {
                    "stage": stage,
                    "epoch": epoch,
                    "training_bce": float(np.mean(losses)),
                    "validation_spearman": metric["mean_spearman"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": time.monotonic() - started,
                }
                history.append(row)
                print(json.dumps(row), flush=True)
                write_json(run_dir / "history.json", history)
                if metric["mean_spearman"] > best_score:
                    best_score, stale = metric["mean_spearman"], 0
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "config": config,
                            "encoder_config": model.encoder.config.to_dict(),
                            "validation_indices": valid_idx.tolist(),
                            "input_sha256": manifest["input_sha256"]["train.csv"],
                        },
                        run_dir / "model.pt",
                    )
                    write_json(run_dir / "metrics.json", metric)
                    write_predictions(
                        run_dir / "validation.csv", frame.qa_id.iloc[valid_idx], prediction
                    )
                elif stage == "fine_tune":
                    stale += 1
                if stage == "fine_tune" and stale >= config["patience"]:
                    break
        del optimizer, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model, _ = load_model(run_dir, device)
        test_prediction = probabilities(model, test_set, config["batch_size"], device)
        # Align explicitly to the official sample order, including target columns.
        write_predictions(run_dir / "submission.csv", test.qa_id, test_prediction)
        import pandas as pd

        submission = pd.read_csv(run_dir / "submission.csv").set_index("qa_id")
        submission.loc[sample.qa_id, sample.columns[1:]].to_csv(run_dir / "submission.csv")
        manifest.update(
            status="completed",
            train_rows=len(train_idx),
            validation_rows=len(valid_idx),
            test_rows=len(test),
            best_validation_spearman=best_score,
            elapsed_seconds=time.monotonic() - started,
        )
        manifest["artifacts"] = {
            str(p.relative_to(run_dir)): sha256(p)
            for p in sorted(run_dir.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        }
        write_json(run_dir / "manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update(
            status="failed",
            error_type=type(error).__name__,
            elapsed_seconds=time.monotonic() - started,
        )
        write_json(run_dir / "manifest.json", manifest)
        raise


def read_sample(path, expected_ids):
    import pandas as pd

    sample = pd.read_csv(path)
    if list(sample.columns) != ["qa_id", *TARGETS] or sample.qa_id.tolist() != expected_ids:
        raise ValueError("Sample submission schema or ID order mismatch")
    return sample


def load_model(run_dir, device):
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    values = checkpoint["encoder_config"].copy()
    kind = values.pop("model_type")
    encoder_config = AutoConfig.for_model(kind, **values)
    encoder = AutoModel.from_config(
        encoder_config, attn_implementation="eager", dtype=torch.float32
    )
    model = QuestModel(checkpoint["config"], encoder=encoder)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def evaluate(data_dir, run_dir, device="cpu"):
    model, checkpoint = load_model(run_dir, torch.device(device))
    configure({**checkpoint["config"], "device": device})
    if sha256(data_dir / "train.csv") != checkpoint["input_sha256"]:
        raise ValueError("Evaluation data differs from checkpoint input")
    frame = read_data(data_dir / "train.csv", labeled=True)
    indices = checkpoint["validation_indices"]
    tokenizer = AutoTokenizer.from_pretrained(run_dir / "tokenizer", local_files_only=True)
    dataset = TokenizedPairs(frame.iloc[indices], tokenizer, checkpoint["config"])
    prediction = probabilities(
        model, dataset, checkpoint["config"]["batch_size"], torch.device(device)
    )
    return mean_spearman(frame[list(TARGETS)].to_numpy()[indices], prediction)


def predict(data_dir, run_dir, output, device="cpu"):
    model, checkpoint = load_model(run_dir, torch.device(device))
    configure({**checkpoint["config"], "device": device})
    frame = read_data(data_dir / "test.csv", labeled=False)
    read_sample(data_dir / "sample_submission.csv", frame.qa_id.tolist())
    tokenizer = AutoTokenizer.from_pretrained(run_dir / "tokenizer", local_files_only=True)
    dataset = TokenizedPairs(frame, tokenizer, checkpoint["config"], labeled=False)
    prediction = probabilities(
        model, dataset, checkpoint["config"]["batch_size"], torch.device(device)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(output, frame.qa_id, prediction)
    receipt = {
        "rows": len(frame),
        "targets": len(TARGETS),
        "test_sha256": sha256(data_dir / "test.csv"),
        "checkpoint_sha256": sha256(run_dir / "model.pt"),
        "output_sha256": sha256(output),
    }
    write_json(output.with_suffix(".receipt.json"), receipt)
    return receipt
