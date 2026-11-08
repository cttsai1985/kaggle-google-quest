"""Question-group CV diagnostics and a bounded, train-only text baseline."""

import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from quest.data import GROUPED, STRATIFIED, TARGETS, make_splits, mean_spearman, read_data, sha256
from quest.runtime import source_identity, write_json, write_predictions


def finite(value):
    return float(value) if np.isfinite(value) else None


def histogram(values):
    values = values[np.isfinite(values)]
    counts, _ = np.histogram(values, bins=[0, 0.25, 0.5, 0.75, 1])
    return counts / counts.sum() if counts.sum() else np.zeros(4)


def analyze(frame, *, folds=5, seed=42):
    target = frame[list(TARGETS)].apply(pd.to_numeric, errors="coerce")
    category = frame.category.value_counts(normalize=True).sort_index()
    support = {}
    for column in TARGETS:
        values = target[column]
        modes = values.mode(dropna=True)
        modal = float(modes.iloc[0]) if len(modes) else None
        informative = values.notna() & (values != modal)
        count = int(frame.loc[informative, "question_body"].nunique())
        support[column] = {
            "missing_rows": int(values.isna().sum()),
            "non_finite_rows": int((~np.isfinite(values)).sum()),
            "out_of_range_rows": int(((values < 0) | (values > 1)).sum()),
            "observed_values": int(values.nunique()),
            "modal_value": modal,
            "nonmodal_question_groups": count,
            "enough_nonmodal_groups_for_every_fold": count >= folds,
        }
    result = {
        "rows": len(frame),
        "question_groups": int(frame.question_body.nunique()),
        "folds": folds,
        "seed": seed,
        "stratification": "category domain only",
        "protocol_selection": "Predeclared category stratification; no seed/score search",
        "target_support": support,
        "protocols": {},
    }
    for protocol in (GROUPED, STRATIFIED):
        rows = []
        for fold, (train, valid) in enumerate(make_splits(frame, protocol, folds=folds, seed=seed)):
            valid_category = frame.category.iloc[valid].value_counts(normalize=True)
            category_distance = float(
                0.5 * (valid_category.reindex(category.index, fill_value=0) - category).abs().sum()
            )
            targets = {}
            for column in TARGETS:
                training, validation = target[column].iloc[train], target[column].iloc[valid]
                targets[column] = {
                    "train_mean": finite(training.mean()),
                    "validation_mean": finite(validation.mean()),
                    "validation_std": finite(validation.std()),
                    "validation_unique_values": int(validation.nunique()),
                    "validation_missing": int(validation.isna().sum()),
                    "validation_non_finite": int((~np.isfinite(validation)).sum()),
                    "distribution_total_variation": float(
                        0.5 * np.abs(histogram(validation) - histogram(target[column])).sum()
                    ),
                }
            rows.append(
                {
                    "fold": fold,
                    "training_rows": len(train),
                    "validation_rows": len(valid),
                    "validation_question_groups": int(frame.question_body.iloc[valid].nunique()),
                    "shared_question_groups": 0,
                    "category_total_variation": category_distance,
                    "category_counts": {
                        str(k): int(v) for k, v in frame.category.iloc[valid].value_counts().items()
                    },
                    "constant_targets": [
                        name
                        for name, stats in targets.items()
                        if stats["validation_unique_values"] <= 1
                    ],
                    "target_diagnostics": targets,
                }
            )
        result["protocols"][protocol] = {
            "folds": rows,
            "max_category_total_variation": max(row["category_total_variation"] for row in rows),
            "max_target_distribution_total_variation": max(
                stats["distribution_total_variation"]
                for row in rows
                for stats in row["target_diagnostics"].values()
            ),
            "validation_row_min": min(row["validation_rows"] for row in rows),
            "validation_row_max": max(row["validation_rows"] for row in rows),
        }
    return result


def vectorize_views(training, validation, config):
    def views(frame):
        limit = config["max_characters"]
        question = (frame.question_title + "\n" + frame.question_body).str.slice(0, limit)
        answer = (frame.question_title + "\n" + frame.answer).str.slice(0, limit)
        return question, answer

    tq, ta = views(training)
    vq, va = views(validation)
    vectorizer = TfidfVectorizer(
        max_features=config["max_features"],
        min_df=config["min_df"],
        ngram_range=tuple(config["ngram_range"]),
        sublinear_tf=True,
        dtype=np.float32,
    )
    # The vocabulary and IDF are fitted within this fold, never on validation text.
    vectorizer.fit(pd.concat([tq, ta], ignore_index=True))
    x_train = hstack([vectorizer.transform(tq), vectorizer.transform(ta)], format="csr")
    x_valid = hstack([vectorizer.transform(vq), vectorizer.transform(va)], format="csr")
    return x_train, x_valid, vectorizer


def score_with_coverage(actual, prediction):
    metric = mean_spearman(actual, prediction)
    defined = np.ptp(actual, axis=0) != 0
    values = np.array(list(metric["per_target"].values()))
    return {
        "mean_spearman_30_zero_fallback": metric["mean_spearman"],
        "mean_spearman_defined_targets": float(values[defined].mean()) if defined.any() else None,
        "defined_targets": int(defined.sum()),
        "per_target": metric["per_target"],
    }


def run_cv(data_path: Path, run_dir: Path, config):
    frame = read_data(data_path, labeled=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    provenance = source_identity()
    diagnostics = analyze(frame, folds=config["folds"], seed=config["seed"])
    write_json(run_dir / "split-analysis.json", diagnostics)
    write_json(run_dir / "config.json", config)
    target = frame[list(TARGETS)].to_numpy()
    oof = np.full(target.shape, np.nan, dtype=np.float32)
    assignments = np.full(len(frame), -1, dtype=int)
    history = []
    with threadpool_limits(limits=config["threads"]):
        for fold, (train, valid) in enumerate(
            make_splits(frame, config["protocol"], folds=config["folds"], seed=config["seed"])
        ):
            x_train, x_valid, vectorizer = vectorize_views(
                frame.iloc[train], frame.iloc[valid], config
            )
            model = Ridge(
                alpha=config["alpha"],
                solver="lsqr",
                tol=config["tolerance"],
                max_iter=config["max_iterations"],
            )
            model.fit(x_train, target[train])
            prediction = np.clip(model.predict(x_valid), 0, 1)
            oof[valid], assignments[valid] = prediction, fold
            score = score_with_coverage(target[valid], prediction)
            row = {
                "fold": fold,
                "training_rows": len(train),
                "validation_rows": len(valid),
                "vocabulary_size": len(vectorizer.vocabulary_),
                **{k: v for k, v in score.items() if k != "per_target"},
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    if not np.isfinite(oof).all() or np.any(assignments < 0):
        raise ValueError("Incomplete out-of-fold predictions")
    pooled = score_with_coverage(target, oof)
    primary = [row["mean_spearman_30_zero_fallback"] for row in history]
    defined = [row["mean_spearman_defined_targets"] for row in history]
    result = {
        "schema_version": 1,
        "project_id": "quest",
        "status": "completed",
        "model": "train-only TF-IDF + Ridge; not transformer cross-validation",
        "source": provenance,
        "input_sha256": sha256(data_path),
        "config": config,
        "environment": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn")
        },
        "fold_metrics": history,
        "pooled_oof": pooled,
        "fold_mean_30_zero_fallback": float(np.mean(primary)),
        "fold_std_30_zero_fallback": float(np.std(primary, ddof=1)),
        "fold_mean_defined_targets": float(np.mean(defined)),
        "fold_std_defined_targets": float(np.std(defined, ddof=1)),
        "dispersion_interpretation": "Descriptive fold SD; not a confidence interval",
        "elapsed_seconds": time.monotonic() - started,
    }
    write_predictions(run_dir / "oof-predictions.csv", frame.qa_id, oof)
    frame[["qa_id"]].assign(fold=assignments).to_csv(run_dir / "fold-assignments.csv", index=False)
    result["artifacts"] = {
        name: sha256(run_dir / name)
        for name in (
            "fold-assignments.csv",
            "oof-predictions.csv",
            "split-analysis.json",
            "config.json",
        )
    }
    write_json(run_dir / "metrics.json", result)
    return result
