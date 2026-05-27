from __future__ import annotations

import random
import warnings
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sentence_transformers import SentenceTransformer
from sentence_transformers.losses.CachedGISTEmbedLoss import CachedGISTEmbedLoss
from sentence_transformers.losses.CachedMultipleNegativesRankingLoss import CachedMultipleNegativesRankingLoss



class ForwardDecorator:
    def __init__(self, fn, linear_layers: dict[int, nn.Linear]) -> None:
        self.fn = fn
        self.linear_layers = linear_layers
        self.dim = None
        self.cache = []
        self.cache_dim = None
        self.idx = 0

    def set_dim(self, dim) -> None:
        self.dim = dim
        self.idx = 0

    def shrink(self, tensor: Tensor) -> Tensor:
        """截断到指定维度，并进行归一化"""
        tensor_dim = tensor.shape[-1]
        if self.dim > tensor_dim:
            raise ValueError(
                f"Dimension {self.dim} in matryoshka_dims cannot be greater than the model's embedding dimension: {tensor_dim}"
            )
        tensor = tensor[..., :self.dim]  # 截断
        tensor = F.normalize(tensor, p=2, dim=-1)  # 归一化
        return tensor

    def transform(self, tensor: Tensor) -> Tensor:
        """在截断后的嵌入上应用线性变换"""
        tensor = self.shrink(tensor)  # 先截断
        tensor = self.linear_layers[str(self.dim)](tensor)  # 应用线性层变换
        return F.normalize(tensor, p=2, dim=-1)  # 再次归一化

    def __call__(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        # Growing cache:
        if self.cache_dim is None or self.cache_dim == self.dim:
            output = self.fn(features)
            self.cache.append(output)
            self.cache_dim = self.dim
        # Using cache:
        else:
            output = self.cache[self.idx]
        if "token_embeddings" in output:
            output["token_embeddings"] = self.transform(output["token_embeddings"])
        output["sentence_embedding"] = self.transform(output["sentence_embedding"])
        self.idx += 1
        return output


class MatryoshkaLinearLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        loss: nn.Module,
        matryoshka_dims: list[int],
        matryoshka_weights: list[float | int] | None = None,
        n_dims_per_step: int = -1,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss = loss
        if isinstance(loss, CachedMultipleNegativesRankingLoss):
            warnings.warn("MatryoshkaLoss is not compatible with CachedMultipleNegativesRankingLoss.", stacklevel=2)
        if isinstance(loss, CachedGISTEmbedLoss):
            warnings.warn("MatryoshkaLoss is not compatible with CachedGISTEmbedLoss.", stacklevel=2)

        if matryoshka_weights is None:
            matryoshka_weights = [1] * len(matryoshka_dims)
        dims_weights = zip(matryoshka_dims, matryoshka_weights)
        self.matryoshka_dims, self.matryoshka_weights = zip(*sorted(dims_weights, key=lambda x: x[0], reverse=True))
        self.n_dims_per_step = n_dims_per_step

        # Define separate Linear layers for each dimension
        self.linear_layers = nn.ModuleDict({
            str(dim): nn.Linear(dim, dim) for dim in self.matryoshka_dims
        })

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        original_forward = self.model.forward
        try:
            decorated_forward = ForwardDecorator(original_forward, self.linear_layers)
            self.model.forward = decorated_forward

            dim_indices = range(len(self.matryoshka_dims))
            if self.n_dims_per_step > 0 and self.n_dims_per_step < len(dim_indices):
                dim_indices = random.sample(dim_indices, self.n_dims_per_step)

            loss = 0.0
            for idx in dim_indices:
                dim = self.matryoshka_dims[idx]
                weight = self.matryoshka_weights[idx]
                decorated_forward.set_dim(dim)
                loss += weight * self.loss(sentence_features, labels)[0]
        finally:
            self.model.forward = original_forward
        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss.__class__.__name__,
            "matryoshka_dims": self.matryoshka_dims,
            "matryoshka_weights": self.matryoshka_weights,
            "n_dims_per_step": self.n_dims_per_step,
        }
