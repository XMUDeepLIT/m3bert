from __future__ import annotations

import csv
import logging
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.readers import InputExample
from sentence_transformers.similarity_functions import SimilarityFunction
import faiss
import torch
from tqdm import tqdm
import hashlib
import json

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)


class MRRNPEnsembleEvaluator(SentenceEvaluator):
    def __init__(
        self,
        sentences1: list[str],
        sentences2: list[str],
        labels: list[float],
        batch_size: int = 1024,
        main_similarity: str | SimilarityFunction | None = None,
        similarity_fn_names: list[Literal["cosine", "euclidean", "manhattan", "dot"]] | None = None,
        name: str = "",
        show_progress_bar: bool = False,
        write_csv: bool = True,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] | None = None,
        truncate_dim: int | None = None,
        ensemble_dims: list[int] = [64, 128],
        ensemble_weights: list[float] = [1, 1],
        cache_dir: str = "/mnt/yx/emb/cache",
        use_cache: bool = True,
        model_name: str = "model"
    ):
        super().__init__()
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels
        self.write_csv = write_csv
        self.precision = precision
        self.truncate_dim = truncate_dim
        self.ensemble_dims = ensemble_dims
        self.ensemble_weights = ensemble_weights 
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.model_name = model_name.replace('/','-')

        assert len(self.ensemble_dims) == len(self.ensemble_weights), "ensemble_dims and ensemble_weights must match in length"
        # assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.labels)

        self.main_similarity = SimilarityFunction(main_similarity) if main_similarity else None
        self.similarity_fn_names = similarity_fn_names or []
        self.name = name

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file = (
            "similarity_evaluation"
            + ("_" + name if name else "")
            + ("_" + precision if precision else "")
            + "_results.csv"
        )
        self.csv_headers = [
            "epoch",
            "steps",
        ]

        self._append_csv_headers(self.similarity_fn_names)

    def _append_csv_headers(self, similarity_fn_names: list[str]) -> None:
        metrics = ["pearson", "spearman"]

        for v in similarity_fn_names:
            for m in metrics:
                self.csv_headers.append(f"{v}_{m}")

    @classmethod
    def from_input_examples(cls, examples: list[InputExample], **kwargs):
        sentences1 = []
        sentences2 = []
        scores = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
        return cls(sentences1, sentences2, scores, **kwargs)

    def _generate_cache_filename(self) -> str:
        """Generate a unique cache filename based on model_name, ensemble_dims, and data sizes."""
        query_count = len(self.sentences1)
        doc_count = len(self.sentences2)
        dims_str = "_".join(map(str, self.ensemble_dims))
        identifier = f"{self.model_name}_{dims_str}_q{query_count}_d{doc_count}"
        identifier_hash = hashlib.md5(identifier.encode()).hexdigest()
        cache_filename = f"{self.model_name}_dims{dims_str}_q{query_count}_d{doc_count}_{identifier_hash}.json"
        return os.path.join(self.cache_dir, cache_filename)


    def __call__(
        self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        cache_file_path = self._generate_cache_filename()
        logger.info(f"MRREvaluator: Evaluating the model on the {self.name} dataset")

        if self.use_cache and os.path.isfile(cache_file_path):
            logger.info(f"Loading cached similarity data from {cache_file_path}")
            with open(cache_file_path, "r") as f:
                cache_data = json.load(f)
        else:
            # todo: 应该用第一个dim找出top100的doc indices之后，doc_indices就固定了，不用每次都去search.
            # 这样也能保证similarities里对应位置的内容是指同一个query-doc pair的相似度
            # 实现上可以先令doc_indices=None,如果是None，就search，然后把所有query对应的len(queries)*100个indices存起来, 否则就直接算这些query与这些indices对应doc的相似度
            cache_data = {}
            top_100_indices = None

            for dim, weight in zip(self.ensemble_dims, self.ensemble_weights):
                logger.info(f"dim={dim} weight={weight}")
                with nullcontext() if dim is None else model.truncate_sentence_embeddings(dim):
                    logger.info(f"encoding 1")
                    embeddings1 = model.encode(
                        self.sentences1,
                        batch_size=self.batch_size,
                        show_progress_bar=self.show_progress_bar,
                        convert_to_numpy=False,
                        precision=self.precision,
                        normalize_embeddings=bool(self.precision),
                    )
                    logger.info(f"encoding 2")
                    embeddings2 = model.encode(
                        self.sentences2,
                        batch_size=self.batch_size,
                        show_progress_bar=self.show_progress_bar,
                        convert_to_numpy=False,
                        precision=self.precision,
                        normalize_embeddings=bool(self.precision),
                    )

                if self.precision == "binary":
                    embeddings1 = (embeddings1 + 128).astype(np.uint8)
                    embeddings2 = (embeddings2 + 128).astype(np.uint8)
                if self.precision in ("ubinary", "binary"):
                    embeddings1 = np.unpackbits(embeddings1, axis=1)
                    embeddings2 = np.unpackbits(embeddings2, axis=1)

                embeddings1 = torch.nn.functional.normalize(torch.stack(embeddings1, dim=0), p=2, dim=-1).cpu()
                embeddings2 = torch.nn.functional.normalize(torch.stack(embeddings2, dim=0), p=2, dim=-1).cpu()
                queries = embeddings1.numpy()
                documents = embeddings2.numpy()

                if top_100_indices is None:
                    # 第一次执行搜索，获取 top 100 索引
                    logger.info(f"faiss.IndexFlatIP")
                    doc_index = faiss.IndexFlatIP(dim)
                    doc_index.add(documents)
                    logger.info(f"searching")
                    _, top_100_indices = doc_index.search(queries, 100)
                    cache_data["top100_indices"] = top_100_indices.tolist()
                
                logger.info(f"calculating similarities")
                dim_similarities = []
                for i, query in enumerate(tqdm(queries, desc="Calculating similarities")):
                    candidate_docs = documents[top_100_indices[i]]

                    similarities = 1 - paired_cosine_distances(
                        np.repeat(query.reshape(1, -1), candidate_docs.shape[0], axis=0), candidate_docs
                    )
                    dim_similarities.append(similarities.tolist())

                cache_data[str(dim)] = dim_similarities

            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"caching")
            with open(cache_file_path, "w") as f:
                json.dump(cache_data, f)
            logger.info(f"done")

        combined_similarities = []
        for dim, weight in zip(self.ensemble_dims, self.ensemble_weights):
            dim_similarities = cache_data[str(dim)]
            weighted_similarities = [weight * np.array(similarities) for similarities in dim_similarities]
            # print(f"weighted_similarities=\n{weighted_similarities}")
            combined_similarities.append(np.stack(weighted_similarities))
        
        # print(f"combined_similarities={combined_similarities}")
        
        final_scores = np.sum(combined_similarities, axis=0)
        # print(f"final_scores={final_scores}")
        # top_10_relative_indices = np.argsort(final_scores, axis=1)[:, -10:]
        top_10_relative_indices = np.argsort(-final_scores, axis=1)[:, :10]
        # print(f"top_10_relative_indices={top_10_relative_indices}")
        top100_indices = np.array(cache_data["top100_indices"])
        top_10_indices = np.take_along_axis(top100_indices, top_10_relative_indices, axis=1)
        # print(f"top_10_indices={top_10_indices}")
        mrr_sum = 0
        for i, (query, label) in enumerate(zip(self.sentences1, self.labels)):
            for j, idx in enumerate(top_10_indices[i]):
                if self.sentences2[idx] == label:
                    mrr_sum += 1.0 / (j + 1)
                    break

        metrics = {f"ensemble_mrr": mrr_sum / len(self.sentences1)}
        logger.info(f"Ensemble MRR: {metrics['ensemble_mrr']:.4f}")

        return metrics


    @property
    def description(self) -> str:
        return "Semantic Similarity"
