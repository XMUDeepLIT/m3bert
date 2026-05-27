import logging
import sys
import traceback
from datetime import datetime
import argparse
import time
from datasets import load_dataset, load_from_disk
import os
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import SequentialEvaluator, SimilarityFunction, MRRMultiGPUEvaluator
from sentence_transformers.training_args import BatchSamplers
import torch.distributed as dist

parser = argparse.ArgumentParser(description="Run Sentence Transformer evaluation.")
parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint directory.")
parser.add_argument("--config_path", type=str, default=None, help="Path to configuration files.")
parser.add_argument("--tokenizer_name_or_path", type=str, required=True, help="Path to tokenizer.")
parser.add_argument("--sleep_min", type=int, default=0, help="Minutes to sleep before starting the evaluation.")
parser.add_argument("--query_size", type=int, default=None, help="Minutes to sleep before starting the evaluation.")
parser.add_argument("--pool_size", type=int, default=None, help="Minutes to sleep before starting the evaluation.")
parser.add_argument("--mrl_dim", type=int, default=64, help="Minutes to sleep before starting the evaluation.")
parser.add_argument("--len1", type=int, default=16, help="Minutes to sleep before starting the evaluation.")
parser.add_argument("--len2", type=int, default=64, help="Minutes to sleep before starting the evaluation.")
parser.add_argument("--layers", type=int, default=4, help="Minutes to sleep before starting the evaluation.")

args = parser.parse_args()


logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

batch_size = 128 
num_train_epochs = 1
matryoshka_dims = [args.mrl_dim]
len1=args.len1
len2=args.len2

ckpt_path=args.ckpt_path
config_path=args.config_path
tokenizer_name_or_path = args.tokenizer_name_or_path
sleep_min=args.sleep_min
query_size=args.query_size
pool_size=args.pool_size
layers=args.layers



def count_parameters_with_details(model):
    total_params = 0
    trainable_params = 0
    module_details = []
    
    for name, module in model.named_modules():
        if hasattr(module, 'parameters'):
            module_params = sum(p.numel() for p in module.parameters())
            module_trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            module_details.append((name, module_params, module_trainable_params))
            total_params += module_params
            trainable_params += module_trainable_params
    
    print(f"{'Module Name':<40} {'Total Params':<15} {'Trainable Params':<15}")
    print("=" * 70)
    for name, mod_params, mod_trainable_params in module_details:
        print(f"{name:<40} {mod_params:<15} {mod_trainable_params:<15}")

logging.info(f"ckpt_path={ckpt_path}\ndim={matryoshka_dims}")

print(f"程序将暂停{args.sleep_min}分钟")

time.sleep(args.sleep_min * 60)
print(f"程序暂停了{args.sleep_min}分钟")

if args.config_path is not None:
    os.system(f"cp {args.config_path}/* {args.ckpt_path}/")

output_dir = f"/mnt/pretrain/emb/experiments/matryoshka_deep-imps_{ckpt_path.replace('/', '-')}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

model = SentenceTransformer(ckpt_path, similarity_fn_name="cosine",tokenizer_name_or_path=tokenizer_name_or_path, trust_remote_code=True)

model.max_seq_length = max(len1,len2)

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    AutoModelForCausalLM
)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
sep_token = tokenizer.sep_token


try:
    del model[0].auto_model.encoder.layer[layers:]
except Exception as e:
    print(f"delete error")
    print(e)
    del model[0].auto_model.layers[layers:]
    print(f"delete success")
count_parameters_with_details(model)

logging.info(model)

train_dataset = load_dataset("json", data_files="/mnt/pretrain/emb/train_query_ads_10M.jsonl", split="train")
eval_dataset = load_from_disk("/mnt/pretrain/emb/1108_testset_1M")
eval_pool = load_from_disk("/mnt/pretrain/emb/1108_doc_testset_1M_unique_9M")


eval_dataset = eval_dataset.select(range(1000000))

if query_size:
    eval_dataset = eval_dataset.select(range(query_size))
if pool_size:
    eval_pool = eval_pool.select(range(pool_size))
print(len(eval_dataset))
print(len(eval_pool))
# 3. Define our training loss
inner_train_loss = losses.SelectionLoss(model)
train_loss = losses.MatryoshkaLoss(model, inner_train_loss, matryoshka_dims=matryoshka_dims)

eval_dataset = eval_dataset.map(lambda example: {
    'query': example['query'].replace("[SEP]", sep_token),
    'document': example['document'].replace("[SEP]", sep_token)
}, batched=False, num_proc=24) 

eval_pool = eval_pool.map(lambda example: {
    'document': example['document'].replace("[SEP]", sep_token)
}, batched=False, num_proc=24)  
print(f"replaced sep")


args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    fp16=True, 
    bf16=False, 
    batch_sampler=BatchSamplers.NO_DUPLICATES, 

    eval_strategy="no",
    eval_steps=100,
    save_strategy="no",
    save_steps=100,
    save_total_limit=2,
    logging_steps=1,
    run_name="matryoshka-deep-imps", 
    disable_tqdm=True
)

if dist.get_rank() == 0:
    evaluators = []
    test_evaluator = \
        MRRMultiGPUEvaluator(
            sentences1=eval_dataset["query"],
            sentences2=eval_pool["document"],
            labels=eval_dataset["document"],
            main_similarity=SimilarityFunction.COSINE,
            name=f"deep-imps-test",
            truncate_dim=matryoshka_dims,
            len1=16,
            len2=64
        )
    test_evaluator(model)