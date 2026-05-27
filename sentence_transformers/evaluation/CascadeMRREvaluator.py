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


if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)

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


class CascadeMRREvaluator(SentenceEvaluator):
    def __init__(
        self,
        sentences1: list[str],
        sentences2: list[str],
        labels: list[float],
        dim1,
        dim2,
        k1,
        k2,
        batch_size: int = 2048,
        main_similarity: str | SimilarityFunction | None = None,
        similarity_fn_names: list[Literal["cosine", "euclidean", "manhattan", "dot"]] | None = None,
        name: str = "",
        show_progress_bar: bool = False,
        write_csv: bool = True,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] | None = None,
        # truncate_dim: int | None = None,
    ):
        super().__init__()
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels
        self.write_csv = write_csv
        self.precision = precision
        # self.truncate_dim = truncate_dim
        self.dim1 = dim1
        self.dim2 = dim2
        self.k1 = k1
        self.k2 = k2

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
        # if self.truncate_dim is not None:
        #     out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(f"CascadeMRREvaluator: Evaluating the model on the {self.name} dataset{out_txt}:")
        ngpus = faiss.get_num_gpus()
        print("number of GPUs:", ngpus)

        with nullcontext() if self.dim1 is None else model.truncate_sentence_embeddings(self.dim1):
            logger.info(f"encoding 1")
            embeddings1_dim1 = encode_on_multiple_gpus(model, self.sentences1, num_gpus=ngpus, batch_size=self.batch_size)
            # embeddings1_dim1 = model.encode(
            #     self.sentences1,
            #     batch_size=self.batch_size,
            #     show_progress_bar=self.show_progress_bar,
            #     convert_to_numpy=False,
            #     precision=self.precision,
            #     normalize_embeddings=bool(self.precision),
            # )
            logger.info(f"done")
            logger.info(f"encoding 2")
            embeddings2_dim1 = encode_on_multiple_gpus(model, self.sentences2, num_gpus=ngpus, batch_size=self.batch_size)

            # embeddings2_dim1 = model.encode(
            #     self.sentences2,
            #     batch_size=self.batch_size,
            #     show_progress_bar=self.show_progress_bar,
            #     convert_to_numpy=False,
            #     precision=self.precision,
            #     normalize_embeddings=bool(self.precision),
            # )
            logger.info(f"done")
        
        query_embeddings_dim1 = torch.nn.functional.normalize(embeddings1_dim1, p=2, dim=-1).cpu()
        document_embeddings_dim1 = torch.nn.functional.normalize(embeddings2_dim1, p=2, dim=-1).cpu()
        
        # 直接切 不用encoding
        with nullcontext() if self.dim2 is None else model.truncate_sentence_embeddings(self.dim2):
            logger.info(f"encoding 1")
            embeddings1_dim2 = encode_on_multiple_gpus(model, self.sentences1, num_gpus=ngpus, batch_size=self.batch_size)
            
            # embeddings1_dim2 = model.encode(
            #     self.sentences1,
            #     batch_size=self.batch_size,
            #     show_progress_bar=self.show_progress_bar,
            #     convert_to_numpy=False,
            #     precision=self.precision,
            #     normalize_embeddings=bool(self.precision),
            # )
            logger.info(f"done")
            logger.info(f"encoding 2")
            embeddings2_dim2 = encode_on_multiple_gpus(model, self.sentences2, num_gpus=ngpus, batch_size=self.batch_size)

            # embeddings2_dim2 = model.encode(
            #     self.sentences2,
            #     batch_size=self.batch_size,
            #     show_progress_bar=self.show_progress_bar,
            #     convert_to_numpy=False,
            #     precision=self.precision,
            #     normalize_embeddings=bool(self.precision),
            # )
            logger.info(f"done")
            
        query_embeddings_dim2 = torch.nn.functional.normalize(embeddings1_dim2, p=2, dim=-1)
        document_embeddings_dim2 = torch.nn.functional.normalize(embeddings2_dim2, p=2, dim=-1)
        
        res = faiss.StandardGpuResources()
        doc_dim1_index = faiss.IndexFlatIP(self.dim1)
        gpu_doc_dim1_index = faiss.index_cpu_to_gpu(res, 0, doc_dim1_index)
        gpu_doc_dim1_index.add(document_embeddings_dim1)
        logger.info(f"searching 1")
        _, indices_k1 = gpu_doc_dim1_index.search(query_embeddings_dim1, self.k1)

        labels = self.labels
        
        logger.info(f"done")

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
                rr_sum = 0
                recall_count = 0 
                idx_q=0
                # np_sentences2 = np.array(self.sentences2)
                for query_embedding_dim2, label_document, candidate_indices in zip(query_embeddings_dim2, labels, indices_k1):
                    # candidate_documents = np_sentences2[candidate_indices].tolist()
                    idx_q+=1
                    candidate_documents = [self.sentences2[idx] for idx in candidate_indices]
                    candidate_embeddings_dim2 = document_embeddings_dim2[candidate_indices]

                    scores = torch.matmul(query_embedding_dim2, candidate_embeddings_dim2.T)
                    indices_k2 = torch.argsort(scores, descending=True)[:self.k2]
                    
                    # doc_dim2_index = faiss.IndexFlatIP(self.dim2)
                    # gpu_doc_dim2_index = faiss.index_cpu_to_gpu(res, 0, doc_dim2_index)
                    # gpu_doc_dim2_index.add(candidate_embeddings_dim2)
                    # _, indices_k2 = gpu_doc_dim2_index.search(query_embedding_dim2.unsqueeze(0), self.k2)
                    
                    # print(indices_k2.shape)
                    flag = False
                    for j, doc_index in enumerate(indices_k2):

                        if label_document == candidate_documents[doc_index]:
                            rr_sum += 1.0 / (j + 1)
                            # print((j + 1))
                            flag = True
                            break
                    if flag:
                        recall_count+=1
                    if idx_q%1000==0:
                        eval_mrr = rr_sum / idx_q
                        eval_recall = recall_count / idx_q
                        metrics[f"mrr_{fn_name}"] = eval_mrr
                        metrics[f"recall_{fn_name}_10"] = eval_recall
                        
                        logger.info(
                            f"{fn_name.capitalize()}-MRR@10:\t{eval_mrr:.4f}"
                        )
                        logger.info(
                            f"{fn_name.capitalize()}-Recall@10:\t{eval_recall:.4f}"
                        )
                
                eval_mrr = rr_sum / len(self.sentences1)
                eval_recall = recall_count / len(self.sentences1)

                metrics[f"mrr_{fn_name}"] = eval_mrr
                metrics[f"recall_{fn_name}_10"] = eval_recall

                logger.info(
                    f"{fn_name.capitalize()}-MRR@10:\t{eval_mrr:.4f}"
                )
                logger.info(
                    f"{fn_name.capitalize()}-Recall@10:\t{eval_recall:.4f}"
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