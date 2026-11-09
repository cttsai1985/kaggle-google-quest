"""Known rank examples and identity failures for retrospective error mining."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quest.data import TARGETS

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "quest_hard_cases", ROOT / "scripts/analyze_hard_cases.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_known_spearman_permutation_and_tied_rank_gap():
    actual = np.array([[1.0], [2.0], [3.0], [4.0]])
    scores, gaps, defined = audit.rank_diagnostics(actual, np.array([[1.0], [3.0], [2.0], [4.0]]))
    assert scores[0] == pytest.approx(0.8)
    np.testing.assert_allclose(gaps[:, 0], [0, 1 / 3, 1 / 3, 0])
    assert defined.tolist() == [True]
    tied = np.array([[1.0], [1.0], [2.0], [3.0]])
    scores, gaps, _ = audit.rank_diagnostics(tied, np.array([[3.0], [3.0], [2.0], [1.0]]))
    assert scores[0] == pytest.approx(-1.0)
    np.testing.assert_allclose(gaps[:, 0], [2 / 3, 2 / 3, 1 / 3, 1])


def test_constant_target_fallback_and_predeclared_stable_cohort():
    scores, _, defined = audit.rank_diagnostics(np.ones((4, 1)), np.arange(4.0)[:, None])
    assert scores.tolist() == [0.0]
    assert defined.tolist() == [False]
    mask = audit.hardest_mask(np.ones(20))
    assert np.flatnonzero(mask).tolist() == [0, 1]
    with pytest.raises(ValueError, match="finite"):
        audit.rank_diagnostics(np.array([[0.0], [1.0]]), np.array([[0.0], [np.nan]]))


def test_exact_validation_identity_and_question_group_boundary():
    train = pd.DataFrame({"qa_id": [1, 2, 3], "question_body": ["fit", "first", "second"]})
    split = pd.DataFrame({"qa_id": [1, 2, 3], "role": ["train", "validation", "validation"]})
    predictions = pd.DataFrame({"qa_id": [3, 2], **{name: [0.0, 0.0] for name in TARGETS}})
    assert audit.align_predictions(train, predictions, split).qa_id.tolist() == [3, 2]
    with pytest.raises(ValueError, match="unique"):
        audit.align_predictions(train, pd.concat([predictions, predictions.iloc[:1]]), split)
    with pytest.raises(ValueError, match="validation population"):
        audit.align_predictions(train, predictions.iloc[:1], split)
    leaked = train.copy()
    leaked.loc[2, "question_body"] = "fit"
    with pytest.raises(ValueError, match="crosses"):
        audit.align_predictions(leaked, predictions, split)


def test_rare_binary_truth_has_tie_limited_correlation_with_distinct_predictions():
    # Even perfect class ordering cannot reach rho=1 with these observed ties.
    truth = [0.0, 0.0, 0.0, 1.0]
    prediction = [0.0, 1.0, 2.0, 3.0]
    assert audit.rank_tie_ceiling(truth, prediction) == pytest.approx(np.sqrt(3 / 5))
    assert audit.rank_tie_ceiling(truth, prediction[::-1]) == pytest.approx(np.sqrt(3 / 5))
