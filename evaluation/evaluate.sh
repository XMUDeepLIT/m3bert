log_path="output.log"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup torchrun --master_port=20017  evaluation/evaluate_mrl_multi_gpu.py \
    --ckpt_path "" \
    --tokenizer_name_or_path "/mnt/pretrain/models/multilingual-e5-base" \
    --config_path "configs/model_configs/llama-custom-mrl" \
    --sleep_min 0 \
    --mrl_dim 32 \
    --layers 4 \
    >  "${log_path}" 2>&1 &

echo ${log_path}