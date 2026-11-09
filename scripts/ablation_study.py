"""Matched text-field ablations on predeclared question-group split seeds."""

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from quest.cv import score_with_coverage
from quest.data import STRATIFIED, TARGETS, make_splits, read_data, sha256
from quest.runtime import source_identity, write_json, write_predictions

VARIANTS = ("title_only", "title_body", "title_body_answer")


def text_views(frame, variant, limit):
    title = frame.question_title
    question = title if variant == "title_only" else title + "\n" + frame.question_body
    answer = title + "\n" + frame.answer if variant == "title_body_answer" else title
    return question.str.slice(0, limit), answer.str.slice(0, limit)


def score(actual, prediction):
    report = score_with_coverage(actual, prediction)
    values = np.array([report["per_target"][name] for name in TARGETS])
    return {
        **report,
        "question_mean_spearman": float(values[:21].mean()),
        "answer_mean_spearman": float(values[21:].mean()),
    }


def run(data, frozen_plan, output, threads):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    frame = read_data(data, labeled=True)
    target = frame[list(TARGETS)].to_numpy()
    frozen = pd.read_csv(frozen_plan)
    if not np.array_equal(frame.qa_id.to_numpy(), frozen.qa_id.to_numpy()):
        raise ValueError("Frozen plan IDs/order differ from training input")
    config = {
        "folds": 5,
        "split_seeds": [42, 43, 44],
        "protocol": STRATIFIED,
        "max_features": 12000,
        "min_df": 3,
        "ngram_range": [1, 2],
        "max_characters": 24000,
        "alpha": 10.0,
        "tolerance": 0.003,
        "max_iterations": 100,
        "threads": threads,
    }
    write_json(output / "config.json", config)
    receipt = {
        "schema_version": 1,
        "project_id": "quest",
        "status": "running",
        "study": "matched two-view TF-IDF/Ridge text-field ablations",
        "source": source_identity(),
        "script_sha256": sha256(Path(__file__)),
        "input_sha256": sha256(data),
        "frozen_seed42_assignment_sha256": sha256(frozen_plan),
        "config": config,
        "environment": {
            "python": platform.python_version(),
            **{
                package: importlib.metadata.version(package)
                for package in ("numpy", "pandas", "scipy", "scikit-learn")
            },
        },
        "runs": [],
        "interpretation": {
            "variance": (
                "Three predeclared grouped split seeds; descriptive split sensitivity, "
                "not stochastic Ridge training variance or a confidence interval."
            ),
            "controlled_change": (
                "Both text slots, vocabulary budget, model/regularization and folds stay fixed. "
                "Add body only to slot 1, then answer only to slot 2. "
                "Vocabulary/IDF refit on each variant's training views."
            ),
            "scope": (
                "CPU text baseline only; not a transformer ablation or leaderboard improvement."
            ),
        },
    }
    write_json(output / "manifest.json", receipt)
    with threadpool_limits(limits=threads):
        for seed in config["split_seeds"]:
            splits = make_splits(frame, STRATIFIED, folds=5, seed=seed)
            assignments = np.full(len(frame), -1)
            for fold, (fit, valid) in enumerate(splits):
                assignments[valid] = fold
                if set(frame.question_body.iloc[fit]) & set(frame.question_body.iloc[valid]):
                    raise ValueError("Question group crosses fit/validation")
            if seed == 42 and not np.array_equal(assignments, frozen.fold.to_numpy()):
                raise ValueError("Seed42 regenerated plan differs from retained authoritative plan")
            assignment_path = output / f"assignments-{seed}.csv"
            frame[["qa_id"]].assign(fold=assignments).to_csv(assignment_path, index=False)
            for variant in VARIANTS:
                run_started = time.monotonic()
                oof = np.full(target.shape, np.nan, dtype=np.float32)
                rows = []
                for fold, (fit, valid) in enumerate(splits):
                    train_views = text_views(frame.iloc[fit], variant, config["max_characters"])
                    valid_views = text_views(frame.iloc[valid], variant, config["max_characters"])
                    vectorizer = TfidfVectorizer(
                        max_features=config["max_features"],
                        min_df=config["min_df"],
                        ngram_range=tuple(config["ngram_range"]),
                        sublinear_tf=True,
                        dtype=np.float32,
                    )
                    vectorizer.fit(pd.concat(train_views, ignore_index=True))
                    x = hstack([vectorizer.transform(view) for view in train_views], format="csr")
                    xv = hstack([vectorizer.transform(view) for view in valid_views], format="csr")
                    model = Ridge(
                        alpha=config["alpha"],
                        solver="lsqr",
                        tol=config["tolerance"],
                        max_iter=config["max_iterations"],
                    )
                    model.fit(x, target[fit])
                    prediction = np.clip(model.predict(xv), 0, 1)
                    oof[valid] = prediction
                    rows.append(
                        {
                            "fold": fold,
                            "rows": len(valid),
                            "vocabulary_size": len(vectorizer.vocabulary_),
                            "shared_question_groups": 0,
                            **score(target[valid], prediction),
                        }
                    )
                if not np.isfinite(oof).all():
                    raise ValueError("Incomplete OOF predictions")
                oof_path = output / f"oof-{seed}-{variant}.csv"
                write_predictions(oof_path, frame.qa_id, oof)
                result = {
                    "split_seed": seed,
                    "variant": variant,
                    "assignment_sha256": sha256(assignment_path),
                    "oof_sha256": sha256(oof_path),
                    "rows": len(frame),
                    "pooled_oof": score(target, oof),
                    "folds": rows,
                    "elapsed_seconds": time.monotonic() - run_started,
                }
                receipt["runs"].append(result)
                write_json(output / "manifest.json", receipt)
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            "variant": variant,
                            "spearman": result["pooled_oof"]["mean_spearman_30_zero_fallback"],
                            "seconds": result["elapsed_seconds"],
                        }
                    ),
                    flush=True,
                )
    summaries = {}
    for variant in VARIANTS:
        values = [
            r["pooled_oof"]["mean_spearman_30_zero_fallback"]
            for r in receipt["runs"]
            if r["variant"] == variant
        ]
        summaries[variant] = {
            "mean": float(np.mean(values)),
            "sample_sd": float(np.std(values, ddof=1)),
            "min": min(values),
            "max": max(values),
            "seeds": len(values),
        }
    paired = []
    for seed in config["split_seeds"]:
        lookup = {r["variant"]: r for r in receipt["runs"] if r["split_seed"] == seed}
        metric = {
            variant: result["pooled_oof"]["mean_spearman_30_zero_fallback"]
            for variant, result in lookup.items()
        }
        paired.append(
            {
                "split_seed": seed,
                "add_body_delta": metric["title_body"] - metric["title_only"],
                "add_answer_delta": metric["title_body_answer"] - metric["title_body"],
            }
        )
    receipt.update(
        status="completed",
        summaries=summaries,
        paired_deltas=paired,
        elapsed_seconds=time.monotonic() - started,
    )
    write_json(output / "manifest.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--frozen-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    run(args.data, args.frozen_plan, args.output, args.threads)
