import json
import shutil

import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizer

from quest.data import TARGETS, TEXT_COLUMNS, TokenizedPairs, sha256
from quest.model import QuestModel
from quest.runtime import probabilities
from quest.serving import export_model


def test_export_retains_text_processing_target_order_and_offline_checkpoint(tmp_path):
    torch.set_num_threads(1)
    run = tmp_path / "run"
    tokenizer_dir = run / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "title", "body", "answer"])
    )
    tokenizer = BertTokenizer(vocab_file=str(vocab))
    tokenizer.save_pretrained(tokenizer_dir)
    encoder = BertModel(
        BertConfig(
            vocab_size=8,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
    )
    config = {"head_width": 8, "question_length": 8, "answer_length": 8}
    model = QuestModel(config, encoder=encoder).eval()
    torch.save(
        {
            "config": config,
            "encoder_config": encoder.config.to_dict(),
            "state_dict": model.state_dict(),
        },
        run / "model.pt",
    )
    names = ["model.pt", *[str(p.relative_to(run)) for p in tokenizer_dir.iterdir()]]
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "source": {"fixture": True},
                "artifacts": {name: sha256(run / name) for name in names},
            }
        )
    )
    example = pd.DataFrame(
        {"question_title": ["title"], "question_body": ["body"], "answer": ["answer"]}
    )
    dataset = TokenizedPairs(example, tokenizer, config, labeled=False)
    expected = probabilities(model, dataset, 1, torch.device("cpu"))
    output = tmp_path / "package"
    export_model(run, output, example)
    shutil.rmtree(run)
    loaded = mlflow.pyfunc.load_model(str(output))
    actual = loaded.predict(example)
    assert list(actual.columns) == list(TARGETS)
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
    with pytest.raises(Exception, match="missing|Missing"):
        loaded.predict(example.drop(columns=TEXT_COLUMNS[0]))
    checkpoint = next((output / "artifacts").rglob("model.pt"))
    checkpoint.write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        mlflow.pyfunc.load_model(str(output))
