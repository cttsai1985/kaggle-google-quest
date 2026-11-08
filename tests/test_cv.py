import json

import numpy as np
import pandas as pd
import pytest

from quest.cv import STRATIFIED, analyze, make_splits, run_cv, vectorize_views
from quest.data import TARGETS


def grouped_frame():
    rows = []
    for group in range(20):
        for answer in range(2):
            rows.append(
                {
                    "qa_id": group * 2 + answer,
                    "category": f"domain{group % 2}",
                    "question_title": "question title",
                    "question_body": f"body group {group}",
                    "answer": f"answer text {answer}",
                    **{name: (group % 3) / 2 for name in TARGETS},
                }
            )
    frame = pd.DataFrame(rows)
    frame[TARGETS[-1]] = 0.0
    frame.loc[frame.question_body == "body group 0", TARGETS[-1]] = 0.5
    return frame


def test_grouped_stratification_is_deterministic_and_covers_each_row_once():
    frame = grouped_frame()
    first, second = make_splits(frame), make_splits(frame)
    held_out = []
    for (train, valid), (_, repeated) in zip(first, second, strict=True):
        np.testing.assert_array_equal(valid, repeated)
        assert not set(frame.question_body.iloc[train]) & set(frame.question_body.iloc[valid])
        held_out.extend(valid)
    assert sorted(held_out) == list(range(len(frame)))


def test_analyzer_reports_missing_targets_and_impossible_rare_group_coverage(tmp_path):
    frame = grouped_frame()
    frame.loc[0, TARGETS[0]] = np.nan
    report = analyze(frame)
    assert report["target_support"][TARGETS[0]]["missing_rows"] == 1
    assert report["target_support"][TARGETS[-1]]["nonmodal_question_groups"] == 1
    assert not report["target_support"][TARGETS[-1]]["enough_nonmodal_groups_for_every_fold"]
    assert (
        sum(
            fold["target_diagnostics"][TARGETS[0]]["validation_missing"]
            for fold in report["protocols"][STRATIFIED]["folds"]
        )
        == 1
    )
    json.dumps(report, allow_nan=False)
    frame.to_csv(tmp_path / "invalid.csv", index=False)
    with pytest.raises(ValueError, match="Targets"):
        run_cv(tmp_path / "invalid.csv", tmp_path / "run", {})


def test_text_preprocessing_fits_training_rows_only():
    training = pd.DataFrame(
        {
            "question_title": ["trainingonly", "trainingonly"],
            "question_body": ["body content", "body content"],
            "answer": ["answer content", "answer content"],
        }
    )
    validation = pd.DataFrame(
        {
            "question_title": ["validationonly"],
            "question_body": ["unseenword"],
            "answer": ["unseenanswer"],
        }
    )
    config = {"max_characters": 1000, "max_features": 100, "min_df": 1, "ngram_range": [1, 1]}
    train, valid, vocabulary = vectorize_views(training, validation, config)
    assert train.shape[1] == valid.shape[1]
    assert "trainingonly" in vocabulary.vocabulary_
    assert "validationonly" not in vocabulary.vocabulary_
    assert "unseenword" not in vocabulary.vocabulary_


def frozen_fixture(tmp_path):
    from quest.data import sha256

    frame = grouped_frame()
    data = tmp_path / "train.csv"
    frame.to_csv(data, index=False)
    assigned = np.full(len(frame), -1)
    splits = make_splits(frame)
    for fold, (_, valid) in enumerate(splits):
        assigned[valid] = fold
    path = tmp_path / "folds.csv"
    frame[["qa_id"]].assign(fold=assigned).to_csv(path, index=False)
    plan = {
        "path": str(path),
        "sha256": sha256(path),
        "input_sha256": sha256(data),
        "protocol": STRATIFIED,
        "folds": 5,
        "seed": 42,
    }
    return frame, plan, splits


def test_frozen_folds_are_authoritative_without_splitter_recomputation(tmp_path, monkeypatch):
    from quest.data import split_fold

    frame, plan, splits = frozen_fixture(tmp_path)

    def unavailable_splitter(*args, **kwargs):
        raise AssertionError("A frozen plan must not rerun the stratification algorithm")

    monkeypatch.setattr("quest.data.StratifiedGroupKFold", unavailable_splitter)
    train, valid = split_fold(
        frame, protocol=STRATIFIED, frozen_plan=plan, input_sha256=plan["input_sha256"]
    )
    np.testing.assert_array_equal(valid, splits[0][1])
    np.testing.assert_array_equal(train, splits[0][0])


@pytest.mark.parametrize("corruption", ["input", "id", "group"])
def test_frozen_plan_rejects_changed_inputs_ids_and_group_leakage(tmp_path, corruption):
    from pathlib import Path

    from quest.data import frozen_splits, sha256

    frame, plan, _ = frozen_fixture(tmp_path)
    if corruption == "input":
        with pytest.raises(ValueError, match="checksum"):
            frozen_splits(frame, plan, input_sha256="changed-dataset")
        return
    path = Path(plan["path"])
    mapping = pd.read_csv(path)
    if corruption == "id":
        mapping.loc[0, "qa_id"] = 100000
    else:
        mapping.loc[0, "fold"] = (mapping.loc[1, "fold"] + 1) % 5
    mapping.to_csv(path, index=False)
    plan["sha256"] = sha256(path)
    with pytest.raises(ValueError, match="IDs differ" if corruption == "id" else "group leakage"):
        frozen_splits(frame, plan, input_sha256=plan["input_sha256"])
