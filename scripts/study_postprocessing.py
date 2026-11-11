"""Restore the original metric-aware rounder and assess it without Transformer retraining."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_hard_cases import align_predictions
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from quest.cv import score_with_coverage, vectorize_views
from quest.data import STRATIFIED, TARGETS, make_splits, mean_spearman, read_data, sha256
from quest.postprocessing import OptimalRounder, postprocess_predictions
from quest.runtime import source_identity, write_json, write_predictions

METHODS = ("raw", "monotone_square", "rounded_all", "rounded_selective")


def target_frame(frame):
    return frame.set_index("qa_id")[list(TARGETS)]


def fit_predict(training, assessment, config):
    x, xv, _ = vectorize_views(training, assessment, config)
    model = Ridge(
        alpha=config["alpha"],
        solver="lsqr",
        tol=config["tolerance"],
        max_iter=config["max_iterations"],
    )
    model.fit(x, training[list(TARGETS)].to_numpy())
    return np.clip(model.predict(xv), 0, 1).astype(float)


def variants(raw, policy):
    return {
        "raw": raw,
        "monotone_square": raw**2,
        "rounded_all": policy.transform(raw, selective=False),
        "rounded_selective": policy.transform(raw),
    }


def summarize(actual, predictions, fold_records):
    report = {}
    for name, prediction in predictions.items():
        scores = [row["scores"][name]["mean_spearman_30_zero_fallback"] for row in fold_records]
        report[name] = {
            "primary_fold_mean_spearman": float(np.mean(scores)),
            "fold_sd_descriptive": float(np.std(scores, ddof=1)),
            "pooled": score_with_coverage(actual, prediction),
        }
    return {
        "conditions": report,
        "folds": fold_records,
        "primary": "mean outer-fold Spearman, fixed 30-target denominator",
        "fold_dispersion": "Descriptive; not a confidence interval",
    }


def nested_ridge(frame, output):
    config = json.loads(Path("configs/cv.json").read_text()) | {"threads": 3}
    predictions = {name: np.full((len(frame), 30), np.nan) for name in METHODS}
    assignments = np.full(len(frame), -1, dtype=int)
    records = []
    for fold, (fit, assessment) in enumerate(make_splits(frame, STRATIFIED, folds=5, seed=42)):
        training = frame.iloc[fit].reset_index(drop=True)
        holdout = frame.iloc[assessment]
        inner_predictions = np.full((len(training), 30), np.nan)
        for inner_fit, inner_valid in make_splits(training, STRATIFIED, folds=3, seed=43):
            inner_predictions[inner_valid] = fit_predict(
                training.iloc[inner_fit], training.iloc[inner_valid], config
            )
        if not np.isfinite(inner_predictions).all():
            raise ValueError("Incomplete inner OOF predictions")
        calibration_y = target_frame(training)
        calibration_p = pd.DataFrame(inner_predictions, index=calibration_y.index, columns=TARGETS)
        policy = OptimalRounder().fit(calibration_y, calibration_p)
        policy.save(output / f"ridge-policy-fold-{fold}.json")
        raw = pd.DataFrame(
            fit_predict(training, holdout, config), index=holdout.qa_id, columns=TARGETS
        )
        scores = {}
        for name, values in variants(raw, policy).items():
            predictions[name][assessment] = values.to_numpy()
            scores[name] = score_with_coverage(holdout[list(TARGETS)].to_numpy(), values.to_numpy())
        if (
            abs(
                scores["raw"]["mean_spearman_30_zero_fallback"]
                - scores["monotone_square"]["mean_spearman_30_zero_fallback"]
            )
            > 1e-12
        ):
            raise ValueError("Strict monotone control unexpectedly changed Spearman")
        assignments[assessment] = fold
        records.append(
            {
                "fold": fold,
                "training_rows": len(fit),
                "assessment_rows": len(assessment),
                "selected_targets": [
                    key for key, value in policy.state.items() if value["selected"]
                ],
                "scores": scores,
                "policy_sha256": sha256(output / f"ridge-policy-fold-{fold}.json"),
            }
        )
        print(
            json.dumps(
                {
                    "outer_fold": fold,
                    "scores": {k: v["mean_spearman_30_zero_fallback"] for k, v in scores.items()},
                }
            ),
            flush=True,
        )
    for name, values in predictions.items():
        if not np.isfinite(values).all():
            raise ValueError("Incomplete outer OOF population")
        write_predictions(output / f"ridge-{name}-oof.csv", frame.qa_id, values)
    split = pd.DataFrame({"qa_id": frame.qa_id, "fold": assignments})
    split.to_csv(output / "ridge-outer-folds.csv", index=False)
    return {
        "scope": (
            "Strict nested postprocessor assessment on a retrospective fixed "
            "TF-IDF/Ridge recipe; not Transformer CV"
        ),
        "rows": len(frame),
        "outer_folds": 5,
        "inner_folds": 3,
        "config": config,
        "outer_split_sha256": sha256(output / "ridge-outer-folds.csv"),
        **summarize(frame[list(TARGETS)].to_numpy(), predictions, records),
    }


def retained_transformer(frame, run, output):
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Expected completed retained transformer producer")
    for name in ["validation.csv", "split.csv", "submission.csv"]:
        if sha256(run / name) != manifest["artifacts"][name]:
            raise ValueError(f"Changed retained artifact: {name}")
    original = pd.read_csv(run / "validation.csv")
    split = pd.read_csv(run / "split.csv")
    heldout = align_predictions(frame, original, split).reset_index(drop=True)
    y, raw = target_frame(heldout), original.set_index("qa_id")[list(TARGETS)]
    headline = mean_spearman(y.to_numpy(), raw.to_numpy())["mean_spearman"]
    if abs(headline - manifest["best_validation_spearman"]) > 1e-12:
        raise ValueError("Retained transformer headline differs")
    naive = OptimalRounder().fit(y, raw)
    naive_scores = {
        name: score_with_coverage(y.to_numpy(), values.to_numpy())
        for name, values in variants(raw, naive).items()
    }
    naive.save(output / "transformer-same-holdout-policy.json")
    predictions = {name: np.full(raw.shape, np.nan) for name in METHODS}
    records = []
    for fold, (calibration, assessment) in enumerate(
        make_splits(heldout, STRATIFIED, folds=5, seed=42)
    ):
        policy = OptimalRounder().fit(y.iloc[calibration], raw.iloc[calibration])
        policy.save(output / f"transformer-diagnostic-policy-{fold}.json")
        scores = {}
        for name, values in variants(raw.iloc[assessment], policy).items():
            predictions[name][assessment] = values.to_numpy()
            scores[name] = score_with_coverage(y.iloc[assessment].to_numpy(), values.to_numpy())
        records.append(
            {
                "fold": fold,
                "calibration_rows": len(calibration),
                "assessment_rows": len(assessment),
                "selected_targets": [k for k, v in policy.state.items() if v["selected"]],
                "scores": scores,
            }
        )
    # Demonstrate the restored saved-policy prediction path without scoring/submitting it.
    test = pd.read_csv(run / "submission.csv")
    before_hash = sha256(run / "submission.csv")
    example = postprocess_predictions(test, output / "transformer-same-holdout-policy.json")
    example.to_csv(output / "transformer-postprocessed-example.csv", index=False)
    if sha256(run / "submission.csv") != before_hash or not test.qa_id.equals(example.qa_id):
        raise ValueError("Example modified the original predictions or their ID order")
    return {
        "scope": (
            "Retrospective saved-checkpoint diagnostics; checkpoint selection used this "
            "holdout, so this is NOT nested Transformer CV or an unbiased new score"
        ),
        "producer_source": manifest["source"],
        "producer_manifest_sha256": sha256(run / "manifest.json"),
        "validation_sha256": sha256(run / "validation.csv"),
        "retained_raw_spearman": headline,
        "rows": len(y),
        "same_label_fit_evaluation_optimistic": naive_scores,
        "calibration_crossfit_diagnostic": summarize(y.to_numpy(), predictions, records),
        "application_example": {
            "test_rows": len(example),
            "original_untouched": True,
            "sample_order_preserved": True,
            "policy_sha256": sha256(output / "transformer-same-holdout-policy.json"),
            "original_predictions_sha256": before_hash,
            "processed_predictions_sha256": sha256(
                output / "transformer-postprocessed-example.csv"
            ),
            "scored_or_submitted": False,
            "promoted": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--transformer-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = source_identity()
    if not source["git_commit"] or source["git_dirty"] is not False:
        raise ValueError("Commit the fixed protocol/code before the study")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    frame = read_data(args.data, labeled=True)
    manifest = json.loads((args.transformer_run / "manifest.json").read_text())
    if sha256(args.data) != manifest["input_sha256"]["train.csv"]:
        raise ValueError("Retained producer training input differs")
    receipt = {
        "schema_version": 1,
        "source": source,
        "script_sha256": sha256(Path(__file__)),
        "postprocessor_sha256": sha256(Path("src/quest/postprocessing.py")),
        "input_sha256": sha256(args.data),
        "methods": list(METHODS),
        "fixed_rounder": {"iterations": 1000, "seed": 42, "min_gain": -0.001},
        "environment": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scipy", "scikit-learn", "quest-labeling"]
        },
        "status": "running",
    }
    write_json(args.output / "manifest.json", receipt)
    with threadpool_limits(limits=3):
        receipt["nested_ridge"] = nested_ridge(frame, args.output)
        receipt["retained_transformer"] = retained_transformer(
            frame, args.transformer_run, args.output
        )
    receipt.update(status="completed", elapsed_seconds=time.monotonic() - started)
    write_json(args.output / "manifest.json", receipt)
    print(
        json.dumps(
            {
                "nested": {
                    k: v["primary_fold_mean_spearman"]
                    for k, v in receipt["nested_ridge"]["conditions"].items()
                },
                "transformer_diagnostic": {
                    k: v["primary_fold_mean_spearman"]
                    for k, v in receipt["retained_transformer"]["calibration_crossfit_diagnostic"][
                        "conditions"
                    ].items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
