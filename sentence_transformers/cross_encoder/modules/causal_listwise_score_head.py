from __future__ import annotations

import torch

from sentence_transformers.base.modules import Module


class CausalListwiseScoreHead(Module):
    config_keys = ["doc_id_token_ids"]

    def __init__(self, doc_id_token_ids: list[int]):
        super().__init__()
        self.doc_id_token_ids = doc_id_token_ids
        self.num_labels = 1

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        logits = features["causal_logits"][:, -1]
        num_docs = features["num_docs"]  # int, set by caller before forward()
        doc_id_token_ids = self.doc_id_token_ids[:num_docs]

        scores = logits[:, doc_id_token_ids]

        features["scores"] = scores
        return features

    def save(self, output_path) -> None:
        self.save_config(output_path)
