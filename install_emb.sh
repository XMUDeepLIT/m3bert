conda create -n emb python=3.10 -y
source activate emb
pip install -r requirements_emb.txt
pip install flash-attn==2.5.5
pip install -e .
pip install faiss-gpu