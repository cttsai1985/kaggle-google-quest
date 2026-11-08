"""Input contracts and question-group validation splits."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset

QUESTION_TARGETS = (
    "question_asker_intent_understanding",
    "question_body_critical",
    "question_conversational",
    "question_expect_short_answer",
    "question_fact_seeking",
    "question_has_commonly_accepted_answer",
    "question_interestingness_others",
    "question_interestingness_self",
    "question_multi_intent",
    "question_not_really_a_question",
    "question_opinion_seeking",
    "question_type_choice",
    "question_type_compare",
    "question_type_consequence",
    "question_type_definition",
    "question_type_entity",
    "question_type_instructions",
    "question_type_procedure",
    "question_type_reason_explanation",
    "question_type_spelling",
    "question_well_written",
)
ANSWER_TARGETS = (
    "answer_helpful",
    "answer_level_of_information",
    "answer_plausible",
    "answer_relevance",
    "answer_satisfaction",
    "answer_type_instructions",
    "answer_type_procedure",
    "answer_type_reason_explanation",
    "answer_well_written",
)
TARGETS = QUESTION_TARGETS + ANSWER_TARGETS
TEXT_COLUMNS = ("question_title", "question_body", "answer")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_data(path: Path, *, labeled: bool) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"qa_id", "category", *TEXT_COLUMNS, *(TARGETS if labeled else ())}
    if required - set(frame):
        raise ValueError(f"Missing required columns: {sorted(required - set(frame))}")
    if frame["qa_id"].isna().any() or frame["qa_id"].duplicated().any():
        raise ValueError("qa_id must be present and unique")
    if frame[list(TEXT_COLUMNS) + ["category"]].isna().any().any():
        raise ValueError("Text and category values must be present")
    if labeled:
        targets = frame[list(TARGETS)].to_numpy(dtype=np.float32)
        if not np.isfinite(targets).all() or not ((0 <= targets) & (targets <= 1)).all():
            raise ValueError("Targets must be finite values in [0, 1]")
    return frame


def group_labels(frame: pd.DataFrame) -> pd.Series:
    # Match the legacy categorical coding, including its ordering in GroupKFold.
    codes = frame["question_body"].astype("category").cat.codes.astype(str)
    return frame["category"].astype(str).str.cat(codes, sep="_")


def split_fold(frame: pd.DataFrame, fold: int = 0, folds: int = 5):
    if not 0 <= fold < folds:
        raise ValueError("Fold must fall within the configured split count")
    groups = group_labels(frame)
    splits = list(GroupKFold(n_splits=folds).split(frame, groups=groups))
    train, valid = splits[fold]
    if set(groups.iloc[train]) & set(groups.iloc[valid]):
        raise ValueError("Question group leakage across train and validation")
    return train, valid


def mean_spearman(targets: np.ndarray, prediction: np.ndarray) -> dict:
    if targets.shape != prediction.shape or targets.shape[1] != len(TARGETS):
        raise ValueError("Expected aligned arrays with all 30 targets")
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite predictions")
    scores = []
    for actual, estimated in zip(targets.T, prediction.T, strict=True):
        # A constant prediction contributes zero; it is never silently excluded.
        if np.ptp(actual) == 0 or np.ptp(estimated) == 0:
            scores.append(0.0)
        else:
            scores.append(float(spearmanr(actual, estimated).statistic))
    return {
        "mean_spearman": float(np.mean(scores)),
        "per_target": dict(zip(TARGETS, scores, strict=True)),
    }


class TokenizedPairs(Dataset):
    def __init__(self, frame, tokenizer, config, *, labeled=True):
        self.question = tokenizer(
            frame.question_title.tolist(),
            frame.question_body.tolist(),
            max_length=config["question_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.answer = tokenizer(
            frame.question_title.tolist(),
            frame.answer.tolist(),
            max_length=config["answer_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.targets = frame[list(TARGETS)].to_numpy(dtype=np.float32) if labeled else None
        self.rows = len(frame)

    def __len__(self):
        return self.rows

    def __getitem__(self, index):
        question = {key: value[index] for key, value in self.question.items()}
        answer = {key: value[index] for key, value in self.answer.items()}
        return question, answer, self.targets[index] if self.targets is not None else 0
