import copy
import datasets
import logging
import numpy as np
import os
import sys
import torch
import torch.distributed as dist
import transformers
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from accelerate import Accelerator
#  main_process_first, local_main_process_first
from torch import Tensor
from torch.nn.functional import normalize, cross_entropy
from dataclasses import dataclass, field
from sklearn.metrics import roc_auc_score
from typing import Any, Dict, List, Optional
from functools import partial
import torch.distributed as dist
from datasets.distributed import split_dataset_by_node
from datasets import Dataset, DatasetDict, load_dataset
import glob
import datetime

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    AutoModelForCausalLM,
    AutoModelForMaskedLM
)

from transformers.trainer_utils import (
    seed_worker,
)

from datasets import load_dataset
import glob
from datasets import load_dataset, load_from_disk
import wandb

wandb.login(key="key") 

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
set_seed(42)

print(transformers.__file__)

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def prepare_features_retrieval(examples, tokenizer, max_length_query=64,max_length_doc=64, retrieval_sentence_keys=['query', 'document']):
    num_examples = len(examples[retrieval_sentence_keys[0]])
    query = ["" if item is None else item for item in examples[retrieval_sentence_keys[0]]]
    document = ["" if item is None else item for item in examples[retrieval_sentence_keys[1]]]

    # in case we use a different separation token for data scraping
    query = [item.replace("[SEP]", tokenizer.sep_token) for item in query]
    document = [item.replace("[SEP]", tokenizer.sep_token) for item in document]

    padding = "max_length" 
    tokenized_data_query = tokenizer(
        query, padding=padding, max_length=max_length_query, truncation=True
    )
    tokenized_data_document = tokenizer(
        document, padding=padding, max_length=max_length_doc, truncation=True
    )

    result = {
        "query_input_ids": tokenized_data_query["input_ids"],
        "doc_input_ids": tokenized_data_document["input_ids"],
    }
    
    assert all([len(result[key]) == num_examples for key in result])
    return result

@dataclass
class RetrievalDataCollator:
    train_batch_size: int = -1
    do_padding: bool = False
    pad_token_id: int = -100
    max_length_query: int = 64
    max_length_doc: int = 64

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        is_training = "label" not in features[0]
        if is_training:
            batch = self.create_inputs(features)
            batch["labels"] = torch.arange(self.train_batch_size, dtype=torch.long)
        else:
            labels = [features[i].pop("label") for i in range(len(features))]
            batch = self.create_inputs(features)
            batch["labels"] = torch.FloatTensor(labels)
        return batch

    def pad_inputs(self, data, max_length):
        return data + [self.pad_token_id] * (max_length - len(data))

    def create_inputs(self, features):
        # print(f"features={features}")
        query_input_ids = []
        doc_input_ids = []

        for feature in features:
            if self.do_padding:
                query_input_ids.append(self.pad_inputs(feature["query_input_ids"], self.max_length_query))
                doc_input_ids.append(self.pad_inputs(feature["doc_input_ids"], self.max_length_doc))
            else:
                query_input_ids.append(feature["query_input_ids"])
                doc_input_ids.append(feature["doc_input_ids"])

        build_mask_func = lambda x: x.ne(self.pad_token_id).type_as(x)  
        query_input_ids=torch.LongTensor(query_input_ids)
        doc_input_ids=torch.LongTensor(doc_input_ids)
        batch = {
            "query_input_ids": query_input_ids,
            "query_attention_mask": build_mask_func(query_input_ids),
            "doc_input_ids": doc_input_ids,
            "doc_attention_mask": build_mask_func(doc_input_ids),
        }

        return batch

class CustomTrainer(Trainer):
    def __init__(self, *args, loss_type='infoNCE', temperature=0.05, dims=None, dim_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.temperature = temperature
        self.dims = dims if dims is not None else [None]
        self.dim_weights = dim_weights if dim_weights is not None else [1.0] * len(self.dims)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # print(f"inputs={inputs}")

        labels = inputs.pop("labels")

        query_input_ids = inputs["query_input_ids"]
        query_attention_mask = inputs["query_attention_mask"]
        doc_input_ids = inputs["doc_input_ids"]
        doc_attention_mask = inputs["doc_attention_mask"]

        query_outputs = model(input_ids=query_input_ids, attention_mask=query_attention_mask)
        doc_outputs = model(input_ids=doc_input_ids, attention_mask=doc_attention_mask)
        query_embeddings = average_pool(query_outputs.last_hidden_state, query_attention_mask)
        doc_embeddings = average_pool(doc_outputs.last_hidden_state, doc_attention_mask)

        loss = 0.0
        for i, dim in enumerate(self.dims):
            if dim is not None:
                query_sub = query_embeddings[:, :dim]
                doc_sub = doc_embeddings[:, :dim]
            else:
                query_sub = query_embeddings
                doc_sub = doc_embeddings
            query_sub = normalize(query_sub, p=2, dim=-1)
            doc_sub = normalize(doc_sub, p=2, dim=-1)
            similarity_matrix = torch.matmul(query_sub, doc_sub.T)
            similarity_matrix /= self.temperature
            sub_loss = cross_entropy(similarity_matrix, labels)

            loss += self.dim_weights[i] * sub_loss

        return (loss, {"logits": similarity_matrix}) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        train_dataloader = DataLoader(train_dataset, **dataloader_params)
        print(f"train_dataloader={train_dataloader}")
        if hasattr(train_dataloader.dataset, "__len__"):
            total_samples = len(train_dataloader.dataset)
            batch_size = train_dataloader.batch_size
            total_batches = len(train_dataloader)

            print(f"Total samples in dataset: {total_samples}")
            print(f"Batch size: {batch_size}")
            print(f"Total number of batches: {total_batches}")
        else:
            print("Dataset does not have __len__, likely an IterableDataset.")
            for i, batch in enumerate(train_dataloader):
                print(f"Batch {i+1}: {batch}")
                if i >= 0: 
                    break
        return train_dataloader


def get_iterable_dataset_length(dataset):
    length = sum(1 for _ in dataset)  
    return length


def main():
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    num_train_epochs=5
    batch_size=1024
    max_length_query=16
    max_length_doc=64
    temperature=0.05
    layers=4
    learning_rate=5e-4
    dims=[64]
    dim_str=str(dims).replace("[","").replace("]","").replace(",","")
    output_dir=f"/mnt/pretrain/ckpts/cl_mbert_{layers}layer_100m_e{num_train_epochs}_t{temperature}_bs{batch_size}_lr{learning_rate}_{dim_str}-{current_time}"
    
    print(f"output_dir={output_dir}")
    print(f"cpu_count={os.cpu_count()}")
    accelerator = Accelerator()
    print(f"accelerator={accelerator}")
    local_rank=accelerator.local_process_index
    global_rank=accelerator.process_index
    world_size=accelerator.num_processes
    print(f"Accelerator info:")
    print(f" - Local rank: {accelerator.local_process_index}") 
    print(f" - Global rank: {accelerator.process_index}")  
    print(f" - World size: {accelerator.num_processes}") 
    print(f" - Is main process: {accelerator.is_main_process}") 
    tokenizer_path='/mnt/pretrain/models/bert-base-multilingual-cased'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    from transformers import TrainerCallback
    class SaveTokenizerCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            tokenizer.save_pretrained(args.output_dir)
    set_seed(42)
    model_path='/mnt/pretrain/models/bert-base-multilingual-cased'
    with accelerator.local_main_process_first():
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True
        )
    print(f"model0={model}")
    set_seed(42)
    del model.encoder.layer[layers:]

    parquet_files = glob.glob('/mnt/pretrain/trainset_100m/**/*.parquet', recursive=True)
    parquet_files.sort()
    logging.info("parquet_files")
    train_dataset = load_dataset("parquet", data_files=parquet_files, split="train", streaming=False)

    is_streaming = not hasattr(train_dataset, "__len__")

    if is_streaming:
        pass
    else:
        logging.info(f"Before splitting: Length of dataset = {len(train_dataset)}")

    train_dataset = split_dataset_by_node(
        train_dataset,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes
    )

    if is_streaming:
        pass
    else:
        logging.info(f"After splitting: Length of dataset = {len(train_dataset)}")

    overwrite_cache=False

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_drop_last=True,
        warmup_ratio=0.1,
        # fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        eval_strategy="no",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=999,
        logging_steps=10,
        learning_rate=learning_rate,
        run_name=output_dir.split('/')[-1]
    )
    print(f"training_args={training_args}")

    prepare_features_partial = partial(
        prepare_features_retrieval,
        tokenizer=tokenizer,
        max_length_query=max_length_query,
        max_length_doc=max_length_doc,
        retrieval_sentence_keys=['query', 'document']
    )

    train_dataset = train_dataset.map(
        prepare_features_partial,
        remove_columns=train_dataset.column_names,
        batched=True,
        batch_size=batch_size,
        num_proc=24,
        drop_last_batch=True,
        load_from_cache_file=not overwrite_cache,
        desc=f"Running tokenizer on train dataset. process {os.cpu_count()//4}",
    )

    data_collator = RetrievalDataCollator(
        train_batch_size=batch_size,
        do_padding=True,
        pad_token_id=tokenizer.pad_token_id,
        max_length_query=max_length_query,
        max_length_doc=max_length_doc,
    )

    # 6. Create the trainer & start training
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        dims=dims,
        temperature=temperature
    )
    trainer.train()

if __name__ == "__main__":
    main()
