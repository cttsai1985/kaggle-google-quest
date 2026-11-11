"""Spearman-aware target discretization, preserving the original OptimalRounder idea."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def column_spearman(actual, predicted):
    """Average-rank Spearman with an explicit zero for undefined correlations."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if np.ptp(actual) == 0 or np.ptp(predicted) == 0:
        return 0.0
    a, b = rankdata(actual), rankdata(predicted)
    return float(np.corrcoef(a, b)[0, 1])


def _frame(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
        raise ValueError("Expected a nonempty DataFrame with unique target columns")
    if any(not isinstance(name, str) or not name for name in frame.columns):
        raise ValueError("Target column names must be nonempty strings")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Target values and predictions must be finite and within [0, 1]")
    return values


def _binned_score(sorted_prediction, centered_truth_prefix, truth_norm, cuts):
    """Score contiguous rank-tie groups without reranking every candidate's rows."""
    n = len(sorted_prediction)
    # Legacy has K cuts for K values: its last cut is redundant after index clipping.
    edges = np.r_[0, np.searchsorted(sorted_prediction, cuts[:-1], side="left"), n]
    counts = np.diff(edges)
    ranks = (edges[:-1] + 1 + edges[1:]) / 2 - (n + 1) / 2
    prediction_norm = float(np.dot(counts, ranks * ranks))
    if truth_norm == 0 or prediction_norm == 0:
        return 0.0
    sums = np.diff(centered_truth_prefix[edges])
    return float(np.clip(np.dot(ranks, sums) / np.sqrt(truth_norm * prediction_norm), -1, 1))


class OptimalRounder:
    """Fit per-target cut points on calibration labels; transform never reads labels.

    Retains uniform random search and the original clipped digitize mapping. Column
    selection uses calibration data only. No improvement is guaranteed on new rows.
    """

    def __init__(self, iterations=1000, seed=42, min_gain=-0.001):
        if type(iterations) is not int or iterations < 0:
            raise ValueError("iterations must be a nonnegative integer")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if not np.isfinite(min_gain):
            raise ValueError("min_gain must be finite")
        self.iterations, self.seed, self.min_gain = iterations, seed, float(min_gain)
        self.columns = None
        self.state = {}

    def fit(self, actual, predicted):
        y, p = _frame(actual), _frame(predicted)
        if not actual.columns.equals(predicted.columns) or not actual.index.equals(predicted.index):
            raise ValueError("Calibration target columns and row identity/order must match")
        self.columns = actual.columns.tolist()
        self.state = {}
        rng = np.random.RandomState(self.seed)
        for j, name in enumerate(self.columns):
            reference = np.unique(y[:, j])
            cuts = np.linspace(0, 1, len(reference))
            order = np.argsort(p[:, j], kind="stable")
            ranks = rankdata(y[:, j]) - (len(y) + 1) / 2
            prefix = np.r_[0.0, np.cumsum(ranks[order])]
            norm = float(np.dot(ranks, ranks))
            score = _binned_score(p[order, j], prefix, norm, cuts)
            for _ in range(self.iterations):
                candidate = np.sort(rng.rand(len(reference)))
                value = _binned_score(p[order, j], prefix, norm, candidate)
                if value > score:
                    cuts, score = candidate, value
            raw = column_spearman(y[:, j], p[:, j])
            rounded = reference[np.searchsorted(cuts[:-1], p[:, j], side="right")]
            self.state[name] = {
                "values": reference.tolist(),
                "cuts": cuts.tolist(),
                "selected": bool(
                    np.ptp(y[:, j]) > 0 and np.ptp(rounded) > 0 and score - raw > self.min_gain
                ),
                "raw_calibration_spearman": raw,
                "rounded_calibration_spearman": score,
                "calibration_rows": len(y),
            }
        return self

    def transform(self, predicted, *, selective=True):
        _frame(predicted)
        if self.columns is None or predicted.columns.tolist() != self.columns:
            raise ValueError("Prediction target columns/order differ from the fitted policy")
        result = predicted.copy()
        for name in self.columns:
            state = self.state[name]
            if selective and not state["selected"]:
                continue
            index = np.searchsorted(state["cuts"][:-1], predicted[name], side="right")
            result[name] = np.asarray(state["values"])[index]
        return result

    def save(self, path):
        if self.columns is None:
            raise ValueError("Fit the postprocessor before saving")
        payload = {
            "schema_version": 1,
            "method": "legacy-random-search-spearman-rounding",
            "iterations": self.iterations,
            "seed": self.seed,
            "min_gain": self.min_gain,
            "columns": self.columns,
            "state": self.state,
        }
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        if payload.get("schema_version") != 1 or payload.get("method") != (
            "legacy-random-search-spearman-rounding"
        ):
            raise ValueError("Unsupported postprocessor schema/method")
        policy = cls(payload["iterations"], payload["seed"], payload["min_gain"])
        columns, state = payload["columns"], payload["state"]
        if (
            not columns
            or any(not isinstance(name, str) or not name for name in columns)
            or len(set(columns)) != len(columns)
            or set(state) != set(columns)
        ):
            raise ValueError("Invalid postprocessor target schema")
        for entry in state.values():
            values, cuts = np.asarray(entry["values"]), np.asarray(entry["cuts"])
            if (
                len(values) == 0
                or values.ndim != 1
                or cuts.shape != values.shape
                or not np.isfinite(values).all()
                or not np.isfinite(cuts).all()
                or ((values < 0) | (values > 1)).any()
                or ((cuts < 0) | (cuts > 1)).any()
                or (np.diff(values) <= 0).any()
                or (np.diff(cuts) < 0).any()
                or type(entry["selected"]) is not bool
            ):
                raise ValueError("Invalid postprocessor values/cut points")
        policy.columns, policy.state = columns, state
        return policy


def postprocess_predictions(frame, policy):
    """Apply an explicit saved policy to an ID-preserving 30-target prediction frame."""
    from quest.data import TARGETS

    if list(frame.columns) != ["qa_id", *TARGETS]:
        raise ValueError("Expected qa_id followed by the 30 targets in their declared order")
    if frame.qa_id.isna().any() or not frame.qa_id.is_unique:
        raise ValueError("Prediction IDs must be unique and nonmissing")
    if not isinstance(policy, OptimalRounder):
        policy = OptimalRounder.load(policy)
    result = frame.copy()
    result[list(TARGETS)] = policy.transform(frame[list(TARGETS)])
    return result
