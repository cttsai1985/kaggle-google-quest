"""Fixed repeated calibration/assessment checks on the retained QUEST checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_hard_cases import align_predictions
from scipy.stats import rankdata

from quest.cv import score_with_coverage
from quest.data import STRATIFIED, TARGETS, make_splits, sha256
from quest.postprocessing import OptimalRounder, postprocess_predictions
from quest.runtime import source_identity, write_json

SEEDS = (42, 1017, 2027, 3407, 7919)
POLICY_SHA = "27ee88acbc793d99957cd3f7febcc2a2cbac34ea8e960687a99e8ec950ed448d"


def score(y, p):
    return score_with_coverage(np.asarray(y), np.asarray(p))


def run(data, producer, previous, output):
    source = source_identity()
    if source["git_dirty"] is not False:
        raise ValueError("Freeze the code and protocol before running")
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((producer / "manifest.json").read_text())
    if manifest["status"] != "completed" or sha256(data) != manifest["input_sha256"]["train.csv"]:
        raise ValueError("Invalid producer or changed training data")
    for name in ("validation.csv", "split.csv", "submission.csv"):
        if sha256(producer / name) != manifest["artifacts"][name]:
            raise ValueError(f"Changed producer artifact: {name}")
    predictions = pd.read_csv(producer / "validation.csv")
    held = align_predictions(pd.read_csv(data), predictions, pd.read_csv(producer / "split.csv"))
    y = held.set_index("qa_id")[list(TARGETS)]
    raw = predictions.set_index("qa_id")[list(TARGETS)]
    records = []
    for seed in SEEDS:
        assignments = np.full(len(held), -1)
        for fold, (calibration, assessment) in enumerate(
            make_splits(held, STRATIFIED, folds=2, seed=seed)
        ):
            if set(held.question_body.iloc[calibration]) & set(held.question_body.iloc[assessment]):
                raise ValueError("Question group leakage")
            policy = OptimalRounder().fit(y.iloc[calibration], raw.iloc[calibration])
            path = output / f"policy-seed-{seed}-fold-{fold}.json"
            policy.save(path)
            processed = OptimalRounder.load(path).transform(raw.iloc[assessment])
            a, b = (
                score(y.iloc[assessment], raw.iloc[assessment]),
                score(y.iloc[assessment], processed),
            )
            record = {
                "seed": seed,
                "assessment_half": fold,
                "calibration_rows": len(calibration),
                "assessment_rows": len(assessment),
                "shared_question_groups": 0,
                "policy_sha256": sha256(path),
                "selected_targets": [k for k, v in policy.state.items() if v["selected"]],
                "raw": a,
                "selective": b,
                "delta": b["mean_spearman_30_zero_fallback"] - a["mean_spearman_30_zero_fallback"],
            }
            assignments[assessment] = fold
            records.append(record)
            print(
                json.dumps({k: record[k] for k in ("seed", "assessment_half", "delta")}), flush=True
            )
        path = output / f"assignments-seed-{seed}.csv"
        pd.DataFrame({"qa_id": held.qa_id, "assessment_half": assignments}).to_csv(
            path, index=False
        )
        for record in records[-2:]:
            record["assignment_sha256"] = sha256(path)
    old = json.loads((previous / "manifest.json").read_text())
    pooled, aligned = {}, {}
    for method in ("raw", "rounded_selective"):
        values = np.full(raw.shape, np.nan)
        normalized = np.full(raw.shape, np.nan)
        for fold, (calibration, assessment) in enumerate(
            make_splits(held, STRATIFIED, folds=5, seed=42)
        ):
            policy = OptimalRounder.load(previous / f"transformer-diagnostic-policy-{fold}.json")
            if policy.state[TARGETS[0]]["calibration_rows"] != len(calibration):
                raise ValueError("Wrong earlier calibration population")
            p = raw.iloc[assessment] if method == "raw" else policy.transform(raw.iloc[assessment])
            rank = (rankdata(p.to_numpy(), axis=0, method="average") - 0.5) / len(p)
            if (
                abs(
                    score(y.iloc[assessment], p)["mean_spearman_30_zero_fallback"]
                    - score(y.iloc[assessment], rank)["mean_spearman_30_zero_fallback"]
                )
                > 1e-12
            ):
                raise ValueError("Rank normalization changed within-fold Spearman")
            values[assessment], normalized[assessment] = p, rank
        pooled[method], aligned[method] = score(y, values), score(y, normalized)
        expected = old["retained_transformer"]["calibration_crossfit_diagnostic"]["conditions"][
            method
        ]["pooled"]
        if (
            abs(
                pooled[method]["mean_spearman_30_zero_fallback"]
                - expected["mean_spearman_30_zero_fallback"]
            )
            > 1e-12
        ):
            raise ValueError("Earlier pooled result differs")
    policy_path = previous / "transformer-same-holdout-policy.json"
    if sha256(policy_path) != POLICY_SHA:
        raise ValueError("Fixed candidate changed")
    policy = OptimalRounder.load(policy_path)
    test = pd.read_csv(producer / "submission.csv")
    processed = postprocess_predictions(test, policy)
    path = output / "submission.csv"
    processed.to_csv(path, index=False)
    if (
        sha256(path)
        != old["retained_transformer"]["application_example"]["processed_predictions_sha256"]
    ):
        raise ValueError("Candidate predictions differ from canary 3")
    receipt = {
        "status": "completed",
        "source": source,
        "script_sha256": sha256(Path(__file__)),
        "protocol_sha256": sha256(Path("docs/deberta-rounding-protocol.md")),
        "producer_manifest_sha256": sha256(producer / "manifest.json"),
        "input_sha256": sha256(data),
        "validation_sha256": sha256(producer / "validation.csv"),
        "scope": (
            "Selected-checkpoint postprocessor robustness diagnostics, "
            "not independent transformer validation"
        ),
        "rows": len(y),
        "targets": 30,
        "seeds": list(SEEDS),
        "transfers": records,
        "mean_paired_delta": float(np.mean([r["delta"] for r in records])),
        "positive_transfers": sum(r["delta"] > 0 for r in records),
        "seed_mean_deltas": {
            str(s): float(np.mean([r["delta"] for r in records if r["seed"] == s])) for s in SEEDS
        },
        "crossfold_scale_diagnostic": {
            "original_pooled": pooled,
            "within_assessment_rank_percentile_pooled": aligned,
            "uses_labels_for_normalization": False,
            "submission_candidate": False,
        },
        "candidate": {
            "policy_sha256": POLICY_SHA,
            "predictions_sha256": sha256(path),
            "test_rows": len(test),
            "original_sha256": sha256(producer / "submission.csv"),
            "same_holdout_optimistic_score": score(y, policy.transform(raw)),
            "scored_or_submitted": False,
            "promoted": False,
        },
    }
    write_json(output / "summary.json", receipt)
    print(
        json.dumps(
            {k: receipt[k] for k in ("mean_paired_delta", "positive_transfers", "seed_mean_deltas")}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.producer, args.previous, args.output)


if __name__ == "__main__":
    main()
