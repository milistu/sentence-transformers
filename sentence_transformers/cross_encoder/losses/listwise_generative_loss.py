from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.modules.causal_listwise_score_head import (
    LISTWISE_DOC_IDS,
    MAX_LISTWISE_DOCS,
    CausalListwiseScoreHead,
)
from sentence_transformers.util import batch_to_device

logger = logging.getLogger(__name__)


class ListwiseGenerativeLoss(nn.Module):
    def __init__(self, model: CrossEncoder) -> None:
        super().__init__()
        self.model = model
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        if not isinstance(model[-1], CausalListwiseScoreHead):
            raise ValueError(
                "ListwiseGenerativeLoss expects a model with a CausalListwiseScoreHead, but got a model with a different head."
            )

    def forward(
        self,
        inputs: list[list[str], list[list[str]]],
        labels: list[Tensor],
        prompt: str | None = None,
        task: str | None = None,
    ) -> Tensor:
        """
        Compute Listwise Generative Loss for a batch of queries and their documents.

        Args:
            inputs: List of (queries, documents_list)
            labels: Ground truth relevance scores, shape (batch_size, num_documents)
            prompt: Prompt to use for the model
            task: Task to use for the model

        Returns:
            Tensor: Mean Listwise Generative Loss over the batch
        """
        if isinstance(labels, Tensor):
            raise ValueError(
                "ListwiseGenerativeLoss expects a list of labels for each sample, but got a single value for each sample."
            )

        if len(inputs) != 2:
            raise ValueError(
                f"ListwiseGenerativeLoss expects two inputs (queries, documents_list), but got {len(inputs)} inputs."
            )

        queries, docs_list = inputs

        if len(queries) != len(labels):
            raise ValueError(f"Number of queries ({len(queries)}) does not match number of labels ({len(labels)}).")

        losses = []
        for query, docs, query_labels in zip(queries, docs_list, labels):
            num_docs = len(docs)

            if num_docs > MAX_LISTWISE_DOCS:
                logger.warning(
                    f"ListwiseGenerativeLoss received {num_docs} documents for a query, "
                    f"but only supports {MAX_LISTWISE_DOCS}. Truncating."
                )
                docs = docs[:MAX_LISTWISE_DOCS]
                query_labels = query_labels[:MAX_LISTWISE_DOCS]
                num_docs = MAX_LISTWISE_DOCS

            letters = LISTWISE_DOC_IDS[:num_docs]

            message = []
            if prompt is not None:
                message.append({"role": "system", "content": prompt})
            message.append({"role": "query", "content": query})
            for letter, doc in zip(letters, docs):
                message.append({"role": f"document_{letter}", "content": doc})

            features = self.model.preprocess([message], task=task)
            features = batch_to_device(features, self.model.device)
            features["num_docs"] = num_docs
            out_features = self.model(features)

            scores = out_features["scores"][0]  # shape: (num_docs,)

            loss = self.cross_entropy_loss(
                scores.unsqueeze(0),  # (1, num_docs)
                query_labels.float().to(self.model.device).softmax(dim=0).unsqueeze(0),  # (1, num_docs)
            )
            losses.append(loss)

        return torch.stack(losses).mean()

    @property
    def citation(self) -> str:
        return """
@inproceedings{cao2007learning,
    title={Learning to Rank: From Pairwise Approach to Listwise Approach},
    author={Cao, Zhe and Qin, Tao and Liu, Tie-Yan and Tsai, Ming-Feng and Li, Hang},
    booktitle={Proceedings of the 24th international conference on Machine learning},
    pages={129--136},
    year={2007}
}
"""
