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

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)


class MRRNPEvaluator(SentenceEvaluator):
    def __init__(
        self,
        sentences1: list[str],
        sentences2: list[str],
        labels: list[float],
        batch_size: int = 2048,
        main_similarity: str | SimilarityFunction | None = None,
        similarity_fn_names: list[Literal["cosine", "euclidean", "manhattan", "dot"]] | None = None,
        name: str = "",
        show_progress_bar: bool = False,
        write_csv: bool = True,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] | None = None,
        truncate_dim: int | None = None,
        ensemble_dims: list[int] = None,
        ensemble_weights: list[float] = None,
        cache_die: str = "/mnt/pretrain/emb/cache"
    ):
        super().__init__()
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels
        self.write_csv = write_csv
        self.precision = precision
        self.truncate_dim = truncate_dim
        self.ensemble_dims = ensemble_dims or []
        self.ensemble_weights = ensemble_weights or []

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

    def __call__(
        self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(f"MRREvaluator: Evaluating the model on the {self.name} dataset{out_txt}:")
        

        with nullcontext() if self.truncate_dim is None else model.truncate_sentence_embeddings(self.truncate_dim):
            logger.info(f"encoding 1")
            embeddings1 = model.encode(
                self.sentences1,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=False,
                precision=self.precision,
                normalize_embeddings=bool(self.precision),
            )
            logger.info(f"done")
            logger.info(f"encoding 2")
            embeddings2 = model.encode(
                self.sentences2,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=False,
                precision=self.precision,
                normalize_embeddings=bool(self.precision),
            )
            logger.info(f"done")
        # Binary and ubinary embeddings are packed, so we need to unpack them for the distance metrics
        if self.precision == "binary":
            embeddings1 = (embeddings1 + 128).astype(np.uint8)
            embeddings2 = (embeddings2 + 128).astype(np.uint8)
        if self.precision in ("ubinary", "binary"):
            embeddings1 = np.unpackbits(embeddings1, axis=1)
            embeddings2 = np.unpackbits(embeddings2, axis=1)

        labels = self.labels

        if not self.similarity_fn_names:
            self.similarity_fn_names = [model.similarity_fn_name]
            self._append_csv_headers(self.similarity_fn_names)

        similarity_functions = {
            "cosine": lambda x, y: 1 - paired_cosine_distances(x, y),
            "manhattan": lambda x, y: -paired_manhattan_distances(x, y),
            "euclidean": lambda x, y: -paired_euclidean_distances(x, y),
            "dot": lambda x, y: [np.dot(emb1, emb2) for emb1, emb2 in zip(x, y)],
        }

        metrics = {}
        for fn_name in self.similarity_fn_names:
            if fn_name in similarity_functions:
                queries = torch.nn.functional.normalize(torch.stack(embeddings1, dim=0), p=2, dim=-1).cpu()
                documents = torch.nn.functional.normalize(torch.stack(embeddings2, dim=0), p=2, dim=-1).cpu()
                rr_sum = 0
                # valid_query_num = 0
                d = queries.shape[-1]
                k = 10
                # todo: 实现ensemble的功能，例如init的时候设置ensemble_dims=[64,128], ensemble_weights=[1,1]
                # 用dim=64和dim=128的model先分别找相似度top1000,然后综合二者的相似度(weight1*sim1+weight2*sim2)，找出top10的document，这个1000和10就直接固定写死在这里
                doc_index = faiss.IndexFlatIP(d)
                print(f"documents={documents}")
                documents = documents.numpy()
                queries = queries.numpy()

                logger.info(f"doc_index.add")
                doc_index.add(documents)
                logger.info(f"searching")
                _, indices = doc_index.search(queries, k) # topk
                logger.info(f"done")

                for i, (query, label_document) in tqdm(enumerate(zip(queries, labels)), desc="Evaluating"):
                    for j, doc_index in enumerate(indices[i][:k]):
                        if label_document == self.sentences2[doc_index]:
                            rr_sum += 1.0 / (j + 1)
                            break

                eval_mrr = rr_sum / len(queries)
                metrics[f"mrr_{fn_name}"] = eval_mrr
                logger.info(
                    f"{fn_name.capitalize()}-MRR@10:\t{eval_mrr:.4f}"
                )

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline="", mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow(
                    [
                        epoch,
                        steps,
                    ]
                    + [
                        metrics[f"{fn_name}_{m}"]
                        for fn_name in self.similarity_fn_names
                        for m in ["pearson", "spearman"]
                    ]
                )

        if len(self.similarity_fn_names) > 1:
            metrics["pearson_max"] = max(metrics[f"pearson_{fn_name}"] for fn_name in self.similarity_fn_names)
            metrics["spearman_max"] = max(metrics[f"spearman_{fn_name}"] for fn_name in self.similarity_fn_names)

        if self.main_similarity:
            self.primary_metric = {
                SimilarityFunction.COSINE: "mrr_cosine",
                SimilarityFunction.EUCLIDEAN: "mrr_euclidean",
                SimilarityFunction.MANHATTAN: "mrr_manhattan",
                SimilarityFunction.DOT_PRODUCT: "mrr_dot",
            }.get(self.main_similarity)
        else:
            if len(self.similarity_fn_names) > 1:
                self.primary_metric = "max"
            else:
                self.primary_metric = f"{self.similarity_fn_names[0]}"

        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics)

        return metrics

    @property
    def description(self) -> str:
        return "Semantic Similarity"
