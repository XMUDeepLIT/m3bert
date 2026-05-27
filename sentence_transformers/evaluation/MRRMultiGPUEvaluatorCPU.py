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
from torch.utils.data import DataLoader
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import copy
if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)


def split_data_across_gpus(data, num_gpus):
    chunk_size = len(data) // num_gpus
    splits = [data[i * chunk_size : (i + 1) * chunk_size] for i in range(num_gpus - 1)]
    splits.append(data[(num_gpus - 1) * chunk_size :])

    return splits

def encode_on_single_gpu(model, split_data, gpu_id, batch_size=1024):
    device = torch.device(f"cuda:{gpu_id}")
    model = copy.deepcopy(model)

    with torch.no_grad():
        embeddings = model.encode(
            split_data,
            batch_size=batch_size,
            show_progress_bar=gpu_id==0,
            convert_to_numpy=False,
            device=device
        )

    return torch.stack(embeddings, dim=0)

def encode_on_multiple_gpus(model, data, num_gpus, batch_size=1024):
    data_splits = split_data_across_gpus(data, num_gpus)

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [
            executor.submit(encode_on_single_gpu, model, split_data, gpu_id, batch_size)
            for gpu_id, split_data in enumerate(data_splits)
        ]

        results = [future.result().cpu() for future in futures]

    return torch.cat(results, dim=0)


class MRRMultiGPUEvaluatorCPU(SentenceEvaluator):
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
    ):
        super().__init__()
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels
        self.write_csv = write_csv
        self.precision = precision
        self.truncate_dim = truncate_dim

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
            ngpus = faiss.get_num_gpus()
            print("number of GPUs:", ngpus)

            logger.info(f"encoding 1")
            embeddings1 = encode_on_multiple_gpus(model, self.sentences1, num_gpus=ngpus, batch_size=self.batch_size)
            # embeddings1 = model.encode(
            #     self.sentences1,
            #     batch_size=self.batch_size,
            #     show_progress_bar=self.show_progress_bar,
            #     convert_to_numpy=False,
            #     precision=self.precision,
            #     normalize_embeddings=bool(self.precision),
            # )
            logger.info(f"done")
            logger.info(f"encoding 2")
            embeddings2 = encode_on_multiple_gpus(model, self.sentences2, num_gpus=ngpus, batch_size=self.batch_size)
            # embeddings2 = model.encode(
            #     self.sentences2,
            #     batch_size=self.batch_size,
            #     show_progress_bar=self.show_progress_bar,
            #     convert_to_numpy=False,
            #     precision=self.precision,
            #     normalize_embeddings=bool(self.precision),
            # )
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
                queries = torch.nn.functional.normalize(embeddings1, p=2, dim=-1).cpu()
                documents = torch.nn.functional.normalize(embeddings2, p=2, dim=-1).cpu()
                # valid_query_num = 0
                d = queries.shape[-1]
                mrr_lens=[1000,100,10]
                # mrr_lens=[10]
                k = max(mrr_lens)
                logger.info(f"faiss.StandardGpuResources")

                res = faiss.StandardGpuResources()

                logger.info(f"faiss.IndexFlatIP")

                doc_index = faiss.IndexFlatIP(d)
                
                documents = documents.numpy()
                queries = queries.numpy()

                logger.info(f"faiss.index_cpu_to_all_gpus")
                
                gpu_doc_index = doc_index
                # gpu_doc_index = faiss.index_cpu_to_gpu(res, 0, doc_index)

                logger.info(f"doc_index.add")
                gpu_doc_index.add(documents)
                print(gpu_doc_index.ntotal)
                logger.info(f"searching")
                dists, indices = gpu_doc_index.search(queries, k)
                logger.info(f"done")
                output_list = []
                for mrr_len in mrr_lens:
                    rr_sum = 0
                    recall_count = 0 
                    for i, (query, label_document) in enumerate(zip(queries, labels)):
                        flag = False
                        for j, doc_index in enumerate(indices[i][:mrr_len]):
                            if label_document == self.sentences2[doc_index]:
                                rr_sum += 1.0 / (j + 1)
                                flag = True
                                break
                        if flag:  # 如果在前 mrr_len 个文档中找到 label_document，增加计数
                            recall_count += 1
                        if (i+1)%100000==0:
                            eval_mrr = rr_sum / (i+1)
                            eval_recall = recall_count / (i+1)
                            print(f"{i+1}:\n")
                            metrics[f"mrr_{fn_name}"] = eval_mrr
                            metrics[f"recall_{fn_name}_10"] = eval_recall
                            logger.info(
                                f"\n{fn_name.capitalize()}-MRR-tmp@{mrr_len}:\t{eval_mrr:.4f}\n"
                                f"\n{fn_name.capitalize()}-Recall-tmp@{mrr_len}:\t{eval_recall:.4f}"
                            )
                        
                    eval_mrr = rr_sum / len(queries)
                    eval_recall = recall_count / len(queries)
                    
                    metrics[f"mrr_{fn_name}"] = eval_mrr
                    metrics[f"recall_{fn_name}_{mrr_len}"] = eval_recall

                    logger.info(
                        f"{fn_name.capitalize()}-MRR@{mrr_len}:\t{eval_mrr:.4f}"
                    )
                    logger.info(
                        f"{fn_name.capitalize()}-Recall@{mrr_len}:\t{eval_recall:.4f}"
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