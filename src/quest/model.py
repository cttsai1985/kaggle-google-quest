"""Shared transformer with padding-aware question and answer representations."""

import torch
from torch import nn
from transformers import AutoModel


def masked_mean(embedding, attention_mask):
    weights = attention_mask.unsqueeze(-1).to(embedding.dtype)
    return (embedding * weights).sum(1) / weights.sum(1).clamp_min(1)


class QuestModel(nn.Module):
    def __init__(self, config, encoder=None):
        super().__init__()
        self.encoder = (
            encoder
            if encoder is not None
            else AutoModel.from_pretrained(
                config["model"],
                revision=config.get("model_revision", "main"),
                attn_implementation="eager",
                dtype=torch.float32,
            )
        )
        width = self.encoder.config.hidden_size
        self.output = nn.Sequential(
            nn.Linear(width * 3, config["head_width"]),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(config["head_width"], 30),
        )
        for layer in self.output.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def freeze_encoder(self, frozen=True):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = not frozen

    def forward(self, question, answer):
        question_features = self.encoder(**question).last_hidden_state
        answer_features = self.encoder(**answer).last_hidden_state
        q = masked_mean(question_features, question["attention_mask"])
        a = masked_mean(answer_features, answer["attention_mask"])
        return self.output(torch.cat([q, a, q - a], dim=1))
