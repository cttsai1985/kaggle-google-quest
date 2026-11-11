"""Current-PyTorch inference for the published Bibimorph QUEST RoBERTa state dict.

Architecture/packing reference: oleg-yaroshevskiy/quest_qa_labeling, yorko,
step5_model3_roberta_code/{model,dataset}.py. This adapter is inference-only.
"""

import hashlib
import json
from math import ceil, floor
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PublicRoberta(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.roberta = RobertaModel(config)
        self.layer_weights = nn.Parameter(torch.zeros(13))
        self.classifier = nn.Linear(768, 30)

    def forward(self, ids, masks):
        output = self.roberta(
            input_ids=ids,
            attention_mask=masks,
            token_type_ids=torch.zeros_like(ids),
            output_hidden_states=True,
            return_dict=True,
        )
        states = torch.stack([layer[:, 0, :] for layer in output.hidden_states], dim=-1)
        mixed = (states * self.layer_weights.softmax(dim=0)).sum(-1)
        # Author's five-sample dropout head; dropout is disabled in inference.
        logits = torch.stack([self.classifier(mixed) for _ in range(5)]).mean(0)
        return logits.sigmoid()


def load(weights, tokenizer_dir, device="cpu"):
    config = json.loads((Path(tokenizer_dir) / "config.json").read_text())
    config["num_hidden_layers"] = 12
    config["num_labels"] = 30
    config["output_hidden_states"] = True
    model = PublicRoberta(RobertaConfig(**config))
    state = torch.load(weights, map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    result = model.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    tokenizer = RobertaTokenizer(
        vocab=str(Path(tokenizer_dir) / "vocab.json"),
        merges=str(Path(tokenizer_dir) / "merges.txt"),
    )
    assert len(tokenizer) == 50265 and tokenizer.pad_token_id == 1
    return model.to(device).eval(), tokenizer


def pack(row, tokenizer):
    parts = [tokenizer.tokenize(str(row[c])) for c in ["question_title", "question_body", "answer"]]
    t, q, a = parts
    if sum(map(len, parts)) + 4 > 500:
        nt = min(26, len(t))
        nq = 260 + ceil((26 - nt) / 2)
        na = 210 + floor((26 - nt) / 2)
        if len(a) < na:
            nq += na - len(a)
            na = len(a)
        elif len(q) < nq:
            na += nq - len(q)
            nq = len(q)
        assert nt + nq + na + 4 == 500

        def ends(values, n):
            front = round(n / 2)
            tail = n - front
            # Match the reference's slicing, including Python's zero-tail behavior.
            return values[:front] + values[-tail:]

        t, q, a = t[:nt], ends(q, nq), ends(a, na)
    tokens = (
        [tokenizer.cls_token]
        + t
        + [tokenizer.sep_token] * 2
        + q
        + [tokenizer.sep_token] * 2
        + a
        + [tokenizer.sep_token] * 2
    )
    ids = tokenizer.convert_tokens_to_ids(tokens)[:500]
    # Preserve the checkpoint's original zero padding, although RoBERTa pad_id is 1.
    return ids + [0] * (500 - len(ids)), [1] * len(ids) + [0] * (500 - len(ids))


def predict(model, tokenizer, frame, device="cpu", batch_size=4):
    result = []
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            batch = [
                pack(row, tokenizer) for _, row in frame.iloc[start : start + batch_size].iterrows()
            ]
            ids = torch.tensor([x[0] for x in batch], device=device)
            masks = torch.tensor([x[1] for x in batch], device=device)
            result.append(model(ids, masks).cpu().numpy())
    values = np.concatenate(result)
    assert values.shape == (len(frame), 30) and np.isfinite(values).all()
    assert ((values >= 0) & (values <= 1)).all()
    return values
