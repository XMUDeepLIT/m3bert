# m3bert: A Modern, Multi-lingual, Matryoshka Bidirectional Encoder

This project is developed based on [sentence-transformers](https://github.com/UKPLab/sentence-transformers).

## Environment Setup

```
conda create -n emb python=3.10 -y
source activate emb
pip install -r requirements_emb.txt
pip install flash-attn==2.5.5
pip install -e .
pip install faiss-gpu
```

## Model Architecture

To inspect the model structure, run:

```
python tests/load_model.py
```

## Fine-tuning

To fine-tune the model, run:

```
bash train/cl_mbert.sh
```

## Evaluation

To evaluate the model, run:

```
bash evaluation/evaluate.sh
```
