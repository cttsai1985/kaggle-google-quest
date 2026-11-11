import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from quest.data import TARGETS
from quest.postprocessing import OptimalRounder, _binned_score, postprocess_predictions


def test_fast_search_objective_matches_scipy_including_ties_and_clipped_last_bin():
    from scipy.stats import rankdata

    rng = np.random.default_rng(10)
    for _ in range(30):
        truth = rng.integers(0, 4, 50) / 3
        prediction = rng.integers(0, 12, 50) / 12
        cuts = np.sort(rng.random(4))
        order = np.argsort(prediction, kind="stable")
        ranks = rankdata(truth) - 25.5
        actual = _binned_score(
            prediction[order], np.r_[0, np.cumsum(ranks[order])], np.dot(ranks, ranks), cuts
        )
        mapped = np.minimum(np.digitize(prediction, cuts), 3)
        expected = 0.0 if np.ptp(mapped) == 0 else spearmanr(truth, mapped).statistic
        assert actual == pytest.approx(expected, abs=1e-12)


def test_learned_ties_can_improve_spearman_but_strict_monotone_map_cannot():
    y = pd.DataFrame({"target": [0.0, 0.0, 1.0, 1.0]})
    p = pd.DataFrame({"target": [0.1, 0.2, 0.3, 0.4]})
    policy = OptimalRounder(iterations=1000).fit(y, p)
    rounded = policy.transform(p)
    assert spearmanr(y.target, p.target).statistic < 1
    assert spearmanr(y.target, rounded.target).statistic == pytest.approx(1)
    assert spearmanr(y.target, p.target**2).statistic == spearmanr(y.target, p.target).statistic


def test_rounder_is_reloadable_preserves_constant_targets_and_rejects_tampering(tmp_path):
    y = pd.DataFrame({"target": [0.0, 0.0, 1.0, 1.0], "constant": [0.0] * 4})
    p = pd.DataFrame({"target": [0.1, 0.2, 0.3, 0.4], "constant": [0.1, 0.2, 0.3, 0.4]})
    original = p.copy()
    policy = OptimalRounder().fit(y, p)
    path = tmp_path / "policy.json"
    policy.save(path)
    loaded = OptimalRounder.load(path)
    pd.testing.assert_frame_equal(policy.transform(p), loaded.transform(p))
    pd.testing.assert_frame_equal(p, original)
    pd.testing.assert_series_equal(loaded.transform(p)["constant"], p["constant"])
    with pytest.raises(ValueError, match="columns/order"):
        loaded.transform(p[p.columns[::-1]])
    bad = json.loads(path.read_text())
    bad["state"]["target"]["cuts"][0] = -1
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="cut points"):
        OptimalRounder.load(path)


def test_saved_prediction_path_preserves_ids_and_target_schema(tmp_path):
    y = pd.DataFrame({name: [0.0, 0.0, 1.0, 1.0] for name in TARGETS})
    p = pd.DataFrame({name: [0.1, 0.2, 0.3, 0.4] for name in TARGETS})
    policy = OptimalRounder(iterations=10).fit(y, p)
    source = p.copy()
    source.insert(0, "qa_id", [4, 3, 2, 1])
    path = tmp_path / "policy.json"
    policy.save(path)
    result = postprocess_predictions(source, path)
    assert result.qa_id.tolist() == [4, 3, 2, 1]
    assert result.columns.tolist() == ["qa_id", *TARGETS]
    assert np.isfinite(result[list(TARGETS)]).all().all()
    with pytest.raises(ValueError, match="identity/order"):
        OptimalRounder().fit(y, p.iloc[::-1])


def test_non_string_target_names_are_rejected_before_serialization():
    values = pd.DataFrame({0: [0.0, 1.0]})
    with pytest.raises(ValueError, match="column names"):
        OptimalRounder().fit(values, values)
