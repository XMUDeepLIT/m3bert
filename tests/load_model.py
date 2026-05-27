
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

config_path = 'configs/model_configs/m3bert'
config = AutoConfig.from_pretrained(config_path,trust_remote_code=True)
model = AutoModel.from_config(config,trust_remote_code=True)
print(model)