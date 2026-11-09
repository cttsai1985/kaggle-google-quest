"""Audit retained QUEST predictions without fitting, filtering or changing a score."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from quest.data import ANSWER_TARGETS, QUESTION_TARGETS, TARGETS

COHORT_FRACTION = 0.10


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_diagnostics(actual, predicted):
    """Return zero-fallback correlations and diagnostic absolute normalized rank gaps."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 2 or len(actual) < 2:
        raise ValueError("Expected aligned target/prediction matrices with at least two rows")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Targets and predictions must be finite")
    scores, gaps, defined = [], np.zeros_like(actual), []
    for column in range(actual.shape[1]):
        truth_rank = rankdata(actual[:, column], method="average")
        prediction_rank = rankdata(predicted[:, column], method="average")
        informative = bool(np.ptp(truth_rank) and np.ptp(prediction_rank))
        score = float(np.corrcoef(truth_rank, prediction_rank)[0, 1]) if informative else 0.0
        scores.append(score)
        defined.append(informative)
        gaps[:, column] = np.abs(truth_rank - prediction_rank) / (len(actual) - 1)
    return np.asarray(scores), gaps, np.asarray(defined)


def rank_tie_ceiling(actual, predicted):
    """Maximum correlation by reordering the observed prediction-rank multiset.

    Average-rank ties fix each vector's variance. Rearrangement aligns the sorted
    vectors to maximize covariance. This is a tie diagnostic, not a new metric.
    """
    actual_rank = np.sort(rankdata(np.asarray(actual), method="average"))
    predicted_rank = np.sort(rankdata(np.asarray(predicted), method="average"))
    if not np.ptp(actual_rank) or not np.ptp(predicted_rank):
        return 0.0
    return float(np.corrcoef(actual_rank, predicted_rank)[0, 1])


def hardest_mask(values):
    """Predeclared top 10%; saved prediction row order breaks exact diagnostic ties."""
    values = np.asarray(values, dtype=float)
    count = int(np.ceil(COHORT_FRACTION * len(values)))
    mask = np.zeros(len(values), dtype=bool)
    mask[np.argsort(-values, kind="stable")[:count]] = True
    return mask


def align_predictions(train, predictions, split):
    for name, frame in [("training", train), ("predictions", predictions), ("split", split)]:
        if "qa_id" not in frame or frame.qa_id.isna().any() or not frame.qa_id.is_unique:
            raise ValueError(f"{name} IDs must be unique and nonmissing")
    if list(predictions.columns) != ["qa_id", *TARGETS]:
        raise ValueError("Prediction target schema/order differs from all 30 declared targets")
    if not {"role", "qa_id"}.issubset(split) or set(split.role) != {"train", "validation"}:
        raise ValueError("Expected the retained train/validation split roles")
    if set(split.qa_id) != set(train.qa_id):
        raise ValueError("Saved split does not cover exactly the source training IDs")
    validation = split.loc[split.role.eq("validation"), "qa_id"]
    if set(validation) != set(predictions.qa_id):
        raise ValueError("Prediction IDs do not equal the retained validation population")
    aligned = train.set_index("qa_id").loc[predictions.qa_id].reset_index()
    if not aligned.qa_id.equals(predictions.qa_id.reset_index(drop=True)):
        raise ValueError("Validation identity/order alignment failed")
    grouped = train.set_index("qa_id")["question_body"]
    fit_groups = set(grouped.loc[split.loc[split.role.eq("train"), "qa_id"]])
    if fit_groups & set(aligned.question_body):
        raise ValueError("Repeated question body crosses the retained split")
    return aligned


def analyze(run_dir, train_path):
    run_dir, train_path = Path(run_dir), Path(train_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "completed":
        raise ValueError("Analyze only a completed producer")
    for name in ["validation.csv", "split.csv"]:
        if sha256(run_dir / name) != manifest["artifacts"][name]:
            raise ValueError(f"Retained artifact hash differs: {name}")
    if sha256(train_path) != manifest["input_sha256"]["train.csv"]:
        raise ValueError("Training input differs from the producer snapshot")
    train = pd.read_csv(train_path)
    predictions = pd.read_csv(run_dir / "validation.csv")
    split = pd.read_csv(run_dir / "split.csv")
    aligned = align_predictions(train, predictions, split)
    actual = aligned[list(TARGETS)].to_numpy(dtype=float)
    estimated = predictions[list(TARGETS)].to_numpy(dtype=float)
    if not ((0 <= actual) & (actual <= 1)).all() or not ((0 <= estimated) & (estimated <= 1)).all():
        raise ValueError("QUEST targets and predictions must lie in [0, 1]")
    scores, gaps, defined = rank_diagnostics(actual, estimated)
    headline = float(scores.mean())
    retained_score = float(manifest["best_validation_spearman"])
    if abs(headline - retained_score) > 1e-12:
        raise ValueError("Recomputed retained headline metric differs")
    row_gaps = gaps.mean(axis=1)
    hard = hardest_mask(row_gaps)
    per_target = []
    for column, name in enumerate(TARGETS):
        _, counts = np.unique(actual[:, column], return_counts=True)
        per_target.append(
            {
                "target": name,
                "spearman": float(scores[column]),
                "defined": bool(defined[column]),
                "spearman_ceiling_given_observed_rank_ties": rank_tie_ceiling(
                    actual[:, column], estimated[:, column]
                ),
                "nonmodal_label_rows": int(len(actual) - counts.max()),
                "distinct_truth_values": int(len(counts)),
                "mean_absolute_normalized_rank_gap": float(gaps[:, column].mean()),
                "hard_cohort_mean_gap": float(gaps[hard, column].mean()),
            }
        )
    families = {}
    for name, targets in [("question", QUESTION_TARGETS), ("answer", ANSWER_TARGETS)]:
        columns = [TARGETS.index(target) for target in targets]
        families[name] = {
            "targets": len(columns),
            "mean_spearman": float(scores[columns].mean()),
            "mean_gap": float(gaps[:, columns].mean()),
            "hard_cohort_mean_gap": float(gaps[hard][:, columns].mean()),
            "remaining_mean_gap": float(gaps[~hard][:, columns].mean()),
        }
    categories = []
    for category in sorted(aligned.category.unique()):
        selected = aligned.category.eq(category).to_numpy()
        count, hard_count = int(selected.sum()), int((selected & hard).sum())
        categories.append(
            {
                "category": str(category),
                "rows": count,
                "hard_cohort_rows": hard_count,
                "hard_cohort_rate": hard_count / count,
                "representation_ratio": (hard_count / hard.sum()) / (count / len(hard)),
                "mean_gap": float(row_gaps[selected].mean()),
            }
        )
    lengths = {}
    for column in ["question_title", "question_body", "answer"]:
        chars = aligned[column].fillna("").str.len().to_numpy()
        lengths[column] = {
            "hard_cohort_median_characters": float(np.median(chars[hard])),
            "remaining_median_characters": float(np.median(chars[~hard])),
        }
    return {
        "schema_version": 1,
        "analyzed_at": datetime.now(UTC).isoformat(),
        "scope": "Retained DeBERTa hosted holdout only; no fitting, score filtering or submission",
        "source_sha256": {
            "analysis_script": sha256(__file__),
            "producer_manifest": sha256(manifest_path),
            "train.csv": sha256(train_path),
            "validation.csv": sha256(run_dir / "validation.csv"),
            "split.csv": sha256(run_dir / "split.csv"),
        },
        "population": {
            "validation_qa_pairs": len(actual),
            "question_groups": int(aligned.question_body.nunique()),
            "all_targets": len(TARGETS),
            "defined_targets": int(defined.sum()),
            "exact_ids_and_group_separation_verified": True,
        },
        "headline": {
            "mean_spearman": headline,
            "retained_mean_spearman": retained_score,
            "absolute_metric_delta": abs(headline - retained_score),
        },
        "diagnostic": {
            "name": "mean absolute per-target rank gap divided by n-1",
            "ties": "average ranks",
            "is_additive_spearman_error_decomposition": False,
            "cohort_fraction_declared": COHORT_FRACTION,
            "hard_cohort_rows": int(hard.sum()),
            "actual_cohort_fraction": float(hard.mean()),
            "tie_break": "retained prediction file row order; stable descending diagnostic sort",
            "all_row_mean_gap": float(row_gaps.mean()),
            "hard_cohort_mean_gap": float(row_gaps[hard].mean()),
            "remaining_mean_gap": float(row_gaps[~hard].mean()),
            "hard_cohort_rank_gap_mass_share": float(row_gaps[hard].sum() / row_gaps.sum())
            if row_gaps.sum()
            else None,
            "selection_is_retrospective_description_not_a_deployment_rule": True,
        },
        "target_families": families,
        "per_target": per_target,
        "category_support": categories,
        "text_length_associations": lengths,
        "limitations": [
            "This is one retained selected-checkpoint holdout, not all-fold transformer OOF.",
            "Rank gap is a diagnostic, not an additive Spearman contribution or error probability.",
            "The hard cohort uses observed labels and cannot select unseen production cases.",
            "Category/length associations are not causal; no observations were removed.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output path; retained evidence is not overwritten")
    receipt = analyze(args.run_dir, args.train)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "rows": receipt["population"]["validation_qa_pairs"],
                "mean_spearman": receipt["headline"]["mean_spearman"],
                "hard_cohort_rows": receipt["diagnostic"]["hard_cohort_rows"],
            }
        )
    )


if __name__ == "__main__":
    main()
