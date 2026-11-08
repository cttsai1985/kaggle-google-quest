import json

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizer

from quest.data import TARGETS, group_labels, mean_spearman, read_data, split_fold
from quest.model import QuestModel, masked_mean
from quest.runtime import evaluate, predict, train


def fixture_data(path):
    path.mkdir()
    rows = []
    for i in range(20):
        rows.append(
            {
                "qa_id": i,
                "category": "science",
                "question_title": "question title",
                "question_body": f"question body {i // 2}",
                "answer": "answer text",
                **{name: float(i % 3) / 2 for name in TARGETS},
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(path / "train.csv", index=False)
    test = frame.iloc[:4].drop(columns=list(TARGETS)).copy()
    test["qa_id"] += 100
    test.to_csv(path / "test.csv", index=False)
    pd.DataFrame({"qa_id": test.qa_id, **{name: 0.5 for name in TARGETS}}).to_csv(
        path / "sample_submission.csv", index=False
    )
    return frame


def test_question_groups_do_not_leak(tmp_path):
    frame = fixture_data(tmp_path / "data")
    a, b = split_fold(frame)
    assert not set(group_labels(frame).iloc[a]) & set(group_labels(frame).iloc[b])
    assert not set(frame.question_body.iloc[a]) & set(frame.question_body.iloc[b])


def test_target_contract_and_constant_metric(tmp_path):
    fixture_data(tmp_path / "data")
    frame = read_data(tmp_path / "data/train.csv", labeled=True)
    actual = frame[list(TARGETS)].to_numpy()
    assert mean_spearman(actual, actual)["mean_spearman"] == pytest.approx(1)
    assert mean_spearman(actual, np.zeros_like(actual))["mean_spearman"] == 0
    frame.loc[0, TARGETS[0]] = 2
    frame.to_csv(tmp_path / "invalid.csv", index=False)
    with pytest.raises(ValueError, match="Targets"):
        read_data(tmp_path / "invalid.csv", labeled=True)


def test_masked_pooling_ignores_padding():
    values = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    torch.testing.assert_close(
        masked_mean(values, torch.tensor([[1, 1, 0]])), torch.tensor([[2.0, 3.0]])
    )


def test_full_fixture_training_checkpoint_and_output(tmp_path):
    data_dir = tmp_path / "data"
    fixture_data(data_dir)
    pretrained = tmp_path / "tiny-bert"
    pretrained.mkdir()
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "question",
        "title",
        "body",
        "answer",
        "text",
        *[str(i) for i in range(10)],
    ]
    (pretrained / "vocab.txt").write_text("\n".join(vocabulary) + "\n")
    tokenizer = BertTokenizer(vocab_file=str(pretrained / "vocab.txt"))
    tokenizer.save_pretrained(pretrained)
    encoder = BertModel(
        BertConfig(
            vocab_size=len(vocabulary),
            hidden_size=24,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=48,
        )
    )
    encoder.half().save_pretrained(pretrained)
    config = json.loads(open("configs/deberta.json").read())
    config.update(
        model=str(pretrained),
        model_revision="main",
        head_width=24,
        question_length=12,
        answer_length=16,
        head_epochs=0,
        epochs=1,
        threads=2,
        device="cpu",
        batch_size=4,
    )
    model = QuestModel(config)
    assert model.encoder.config.hidden_size == 24
    assert all(p.dtype == torch.float32 for p in model.parameters())
    from quest.data import make_splits, sha256

    frame = pd.read_csv(data_dir / "train.csv")
    assigned = np.full(len(frame), -1)
    for fold, (_, valid) in enumerate(make_splits(frame)):
        assigned[valid] = fold
    plan = tmp_path / "fold-assignments.csv"
    frame[["qa_id"]].assign(fold=assigned).to_csv(plan, index=False)
    config["fold_plan"] = {
        "path": str(plan),
        "sha256": sha256(plan),
        "input_sha256": sha256(data_dir / "train.csv"),
        "folds": 5,
        "seed": 42,
        "protocol": config["split_protocol"],
    }
    run = tmp_path / "run"
    result = train(data_dir, run, config)
    assert (run / "fold-assignments.csv").read_bytes() == plan.read_bytes()
    plan.unlink()  # Inference must not need the original external CV plan.

    assert result["status"] == "completed"
    assert result["train_rows"] + result["validation_rows"] == 20
    saved = json.loads((run / "metrics.json").read_text())
    assert evaluate(data_dir, run) == saved
    output = tmp_path / "prediction.csv"
    receipt = predict(data_dir, run, output)
    assert receipt["rows"] == 4
    assert pd.read_csv(output).qa_id.tolist() == [100, 101, 102, 103]
    assert pd.read_csv(output).columns.tolist() == ["qa_id", *TARGETS]
    with pytest.raises(FileExistsError):
        train(data_dir, run, config)


def test_source_identity_survives_a_runtime_without_git(monkeypatch):
    from quest.runtime import source_identity

    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr("quest.runtime.subprocess.run", missing_git)
    identity = source_identity()
    assert len(identity["python_source_sha256"]) == 64
    assert (
        identity["python_source_sha256"]
        != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_deberta_checkpoint_reconstructs_without_model_hub(tmp_path):
    from transformers import DebertaV2Config, DebertaV2Model

    from quest.runtime import load_model

    torch.set_num_threads(2)
    encoder = DebertaV2Model(
        DebertaV2Config(
            vocab_size=40,
            hidden_size=24,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=48,
            relative_attention=True,
            pos_att_type=["p2c", "c2p"],
            position_biased_input=False,
        )
    )
    model = QuestModel({"head_width": 24}, encoder=encoder).eval()
    pair = {
        "input_ids": torch.randint(0, 40, (2, 12)),
        "attention_mask": torch.ones(2, 12, dtype=torch.long),
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {"head_width": 24},
            "encoder_config": encoder.config.to_dict(),
        },
        tmp_path / "model.pt",
    )
    reloaded, _ = load_model(tmp_path, torch.device("cpu"))
    with torch.inference_mode():
        torch.testing.assert_close(model(pair, pair), reloaded(pair, pair))
