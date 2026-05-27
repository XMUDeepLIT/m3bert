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
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def split_data_across_gpus(data, num_gpus):
    chunk_size = len(data) // num_gpus
    splits = [data[i * chunk_size : (i + 1) * chunk_size] for i in range(num_gpus - 1)]
    splits.append(data[(num_gpus - 1) * chunk_size :])

    return splits

def encode_on_single_gpu(model, pooler, tokenizer, split_data, gpu_id,  batch_size=1024, max_len=64):
    device = torch.device(f"cuda:{gpu_id}")
    model = copy.deepcopy(model).to(device)
    pooler = copy.deepcopy(pooler).to(device)
    # todo:模型和数据都应该移到device上去，tokenize是不是也能指定device

    show_progress_bar=gpu_id==0
    # Prepare the data split (tokenization)
    # input_ids = []
    # attention_masks = []
    
    # # Tokenize the data (split_data is assumed to be a list of text samples)
    # for text in split_data:
    #     encoding = tokenizer(
    #         text,
    #         truncation=True,
    #         padding='max_length',
    #         max_length=max_len,
    #         return_tensors="pt"
    #     )
    #     input_ids.append(encoding['input_ids'])
    #     attention_masks.append(encoding['attention_mask'])
    
    # # Convert to tensors
    # input_ids = torch.cat(input_ids, dim=0).to(device)
    # attention_masks = torch.cat(attention_masks, dim=0).to(device)
    
    embeddings = []
    
    # Tokenize the data in batches using batch_encode_plus
    for i in tqdm(range(0, len(split_data), batch_size), disable=not show_progress_bar):
        batch_text = split_data[i:i+batch_size]
        
        # Use batch_encode_plus to tokenize the batch
        encoding = tokenizer.batch_encode_plus(
            batch_text,
            truncation=True,
            padding='max_length',
            max_length=max_len,
            return_tensors="pt",
            return_attention_mask=True
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Forward pass through the model
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Pass the model's output through the pooler
            if "average" in str(type(pooler).__name__).lower():
                pooled_embed = pooler(outputs.last_hidden_state, attention_mask)
            else: # attenion pooler
                hidden_state=outputs.last_hidden_state
                extended_attention_mask = model.get_extended_attention_mask(
                    attention_mask, hidden_state.shape
                )
                pooled_embed = pooler(
                    hidden_state, extended_attention_mask,
                    final_pooling_mask=attention_mask,
                )
                pooled_embed = F.tanh(pooled_embed)
            
            # Append the pooled embeddings to the list (on CPU for final aggregation)
            embeddings.append(pooled_embed.cpu())
    
    # Concatenate all the embeddings into a single tensor
    embeddings = torch.cat(embeddings, dim=0)
    
    return embeddings


def encode_on_multiple_gpus(model, pooler, tokenizer, data, num_gpus, batch_size=1024, max_len=64):
    data_splits = split_data_across_gpus(data, num_gpus)

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [
            executor.submit(encode_on_single_gpu, model, pooler, tokenizer, split_data, gpu_id, batch_size, max_len)
            for gpu_id, split_data in enumerate(data_splits)
        ]

        results = [future.result().cpu() for future in futures]

    return torch.cat(results, dim=0)


class MRRMultiGPUEvaluatorTwoTower(SentenceEvaluator):
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
        truncate_dim = None,
        len1=None,
        len2=None,
        tokenizer=None
    ):
        super().__init__()
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels
        self.write_csv = write_csv
        self.precision = precision
        self.truncate_dim = truncate_dim
        self.len1=len1
        self.len2=len2
        self.tokenizer = tokenizer
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
        self, model, output_path: str = None, epoch: int = -1, steps: int = -1
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

        if isinstance(self.truncate_dim,int):
            mrl_dims=[self.truncate_dim]
        else:
            mrl_dims=self.truncate_dim
        max_dim=max(mrl_dims)
        ngpus = faiss.get_num_gpus()
        print("number of GPUs:", ngpus)
        logger.info(f"encoding 1")
        # query_bert
        embeddings1_ori = encode_on_multiple_gpus(
            model.query_model if hasattr(model, 'query_model') else model.query_bert, 
            model.query_pooler, self.tokenizer,
            self.sentences1, num_gpus=ngpus, batch_size=self.batch_size, max_len=self.len1
        )
        logger.info(f"done")
        logger.info(f"encoding 2")
        embeddings2_ori = encode_on_multiple_gpus(
            model.keyword_model if hasattr(model, 'keyword_model') else model.keyword_bert, 
            model.keyword_pooler, self.tokenizer,
            self.sentences2, num_gpus=ngpus, batch_size=self.batch_size, max_len=self.len2
        )
        logger.info(f"done")

        for mrl_dim in mrl_dims:
            print(f"mrl_dim={mrl_dim}")
            embeddings1 = embeddings1_ori[...,:mrl_dim]
            embeddings2 = embeddings2_ori[...,:mrl_dim]
            # Binary and ubinary embeddings are packed, so we need to unpack them for the distance metrics
            if self.precision == "binary":
                embeddings1 = (embeddings1 + 128).astype(np.uint8)
                embeddings2 = (embeddings2 + 128).astype(np.uint8)
            if self.precision in ("ubinary", "binary"):
                embeddings1 = np.unpackbits(embeddings1, axis=1)
                embeddings2 = np.unpackbits(embeddings2, axis=1)

            labels = self.labels

            metrics = {}

            queries = torch.nn.functional.normalize(embeddings1, p=2, dim=-1).cpu().to(dtype=torch.float32)
            documents = torch.nn.functional.normalize(embeddings2, p=2, dim=-1).cpu().to(dtype=torch.float32)
            # valid_query_num = 0
            d = queries.shape[-1]
            mrr_lens=[1000,100,10]
            # mrr_lens=[10]
            k = max(mrr_lens)

            res = faiss.StandardGpuResources()

            doc_index = faiss.IndexFlatIP(d)

            gpu_doc_index = faiss.index_cpu_to_all_gpus(doc_index)

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
                        if label_document[:100] == self.sentences2[doc_index][:100]:
                            rr_sum += 1.0 / (j + 1)
                            flag = True
                            break
                    if flag:  # 如果在前 mrr_len 个文档中找到 label_document，增加计数
                        recall_count += 1
                    if (i+1)%100000==0:
                        eval_mrr = rr_sum / (i+1)
                        eval_recall = recall_count / (i+1)
                        print(f"{i+1}:\n")
                        metrics[f"mrr_{mrr_len}"] = eval_mrr
                        metrics[f"recall_{mrr_len}"] = eval_recall
                        logger.info(
                            f"\n-MRR-tmp@{mrr_len}:\t{eval_mrr:.4f}\n"
                            f"\n-Recall-tmp@{mrr_len}:\t{eval_recall:.4f}"
                        )
                    
                eval_mrr = rr_sum / len(queries)
                eval_recall = recall_count / len(queries)
                
                metrics[f"mrr_{mrr_len}"] = eval_mrr
                metrics[f"recall_{mrr_len}"] = eval_recall

                logger.info(
                    f"-MRR@{mrr_len}:\t{eval_mrr:.4f}"
                )
                logger.info(
                    f"-Recall@{mrr_len}:\t{eval_recall:.4f}"
                )

            if output_path is not None:
                os.makedirs(output_path,exist_ok=True)
                with open(f"{output_path}/results.jsonl", 'w', encoding='utf-8') as out:
                    out.write(json.dumps(metrics) + '\n')
        return metrics

    @property
    def description(self) -> str:
        return "Semantic Similarity"