@echo off

echo [0/6] Installing requirements...
pip install -r requirements.txt

echo [1/6] Creating folders...
mkdir data
mkdir artifacts

echo [2/6] Downloading dataset...
curl -L -o data\dataframe.csv "https://huggingface.co/datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset/resolve/main/bitext-retail-banking-llm-chatbot-training-dataset.csv?download=true"

echo [3/6] Cloning embedder model...
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 tmp_embedder
move tmp_embedder artifacts\embedder

echo [4/6] Cloning correcter model...
git clone https://huggingface.co/oliverguhr/spelling-correction-english-base tmp_correcter
move tmp_correcter artifacts\correcter

echo [5/6] Cloning LLM model...
git clone https://huggingface.co/Qwen/Qwen3-0.6B tmp_llm
move tmp_llm artifacts\llm

echo [6/6] Preparing data and indexes...
python main.py
echo Finished...