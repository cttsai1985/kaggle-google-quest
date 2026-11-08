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
