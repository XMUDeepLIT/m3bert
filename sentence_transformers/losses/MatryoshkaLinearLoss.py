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
from torch.cuda.amp import autocast


class ForwardDecorator:
    def __init__(self, fn, linear_layer: nn.Linear) -> None:
        self.fn = fn
        self.linear_layer = linear_layer
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
        # print(f"tensor_dim={tensor_dim}: {tensor.shape}")
        if self.dim > tensor_dim:
            raise ValueError(
                f"Dimension {self.dim} in matryoshka_dims cannot be greater than the model's embedding dimension: {tensor_dim}"
            )
        tensor = tensor[..., :self.dim]  # 截断
        tensor = F.normalize(tensor, p=2, dim=-1)  # 归一化
        return tensor

    def transform(self, tensor: Tensor) -> Tensor:
        # print(f"shape={tensor.shape}")
        with autocast(dtype=torch.float16):
            # Ensure input and layer weights are in FP16
            tensor = tensor.to(torch.float16)
            tensor = self.linear_layer(tensor)
        # tensor = self.linear_layer(tensor) 
        return  tensor

    def __call__(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        # Growing cache:
        if self.cache_dim is None or self.cache_dim == self.dim:   # cache空和dim不变的情况下，都不能直接用
            output = self.fn(features)
            if "token_embeddings" in output:
                output["token_embeddings"] = self.transform(output["token_embeddings"])
            output["sentence_embedding"] = self.transform(output["sentence_embedding"])
            self.cache.append(output)
            self.cache_dim = self.dim
        # Using cache:     cacbe非空或者dim变化，都用cache
        else:
            output = self.cache[self.idx]
        if "token_embeddings" in output:
            output["token_embeddings"] = self.shrink(output["token_embeddings"])
        output["sentence_embedding"] = self.shrink(output["sentence_embedding"])
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
        model_dim=768,
        output_dim=-1
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
        if output_dim==-1:
            output_dim=model_dim
        self.linear_layer = nn.Linear(model_dim, output_dim).half()

    def transform(self, tensor: Tensor) -> Tensor:
        # print(f"shape={tensor.shape}")
        # with autocast(dtype=torch.float16):
        #     # Ensure input and layer weights are in FP16
        #     tensor = tensor.to(torch.float16)
        #     tensor = self.linear_layer(tensor)
        tensor = self.linear_layer(tensor) 
        return  tensor

    def shrink(self, tensor: Tensor, dim) -> Tensor:
        """截断到指定维度，并进行归一化"""
        tensor_dim = tensor.shape[-1]
        # print(f"tensor_dim={tensor_dim}: {tensor.shape}")
        if dim > tensor_dim:
            raise ValueError(
                f"Dimension {dim} in matryoshka_dims cannot be greater than the model's embedding dimension: {tensor_dim}"
            )
        tensor = tensor[..., :dim]
        tensor = F.normalize(tensor, p=2, dim=-1)
        return tensor

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        original_forward = self.model.forward
        try:
            # decorated_forward = ForwardDecorator(original_forward, self.linear_layer)
            # self.model.forward = decorated_forward

            dim_indices = range(len(self.matryoshka_dims))
            if self.n_dims_per_step > 0 and self.n_dims_per_step < len(dim_indices):
                dim_indices = random.sample(dim_indices, self.n_dims_per_step)

            loss = 0.0
            with autocast(dtype=torch.float16):
                reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

            # reps = [self.transform(self.model(sentence_feature)["sentence_embedding"]) for sentence_feature in sentence_features]
            # # print(f"reps = {reps}")
            embeddings_query, embeddings_document = reps
            for idx in dim_indices:
                dim = self.matryoshka_dims[idx]
                weight = self.matryoshka_weights[idx]
                embeddings_query=self.shrink(embeddings_query,dim)
                embeddings_document=self.shrink(embeddings_document,dim)
                loss += weight * self.loss(embeddings_query,embeddings_document,labels)[0]

            # for idx in dim_indices:
            #     dim = self.matryoshka_dims[idx]
            #     weight = self.matryoshka_weights[idx]
            #     decorated_forward.set_dim(dim)

            #     loss += weight * self.loss(sentence_features, labels)[0]
            #     self.model.forward = original_forward
        finally:
            pass
            # self.model.forward = original_forward
        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss.__class__.__name__,
            "matryoshka_dims": self.matryoshka_dims,
            "matryoshka_weights": self.matryoshka_weights,
            "n_dims_per_step": self.n_dims_per_step,
        }
