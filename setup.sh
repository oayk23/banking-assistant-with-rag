#!/bin/bash

echo "[0/6] Installing Python dependencies..."
pip install -r requirements.txt

echo "[1/6] Creating folders..."
mkdir -p data
mkdir -p artifacts

echo "[2/6] Downloading dataset..."
curl -L "https://huggingface.co/datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset/resolve/main/bitext-retail-banking-llm-chatbot-training-dataset.csv?download=true" -o data/dataframe.csv

echo "[3/6] Cloning embedder model..."
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 tmp_embedder
mv tmp_embedder artifacts/embedder

echo "[4/6] Cloning correcter model..."
git clone https://huggingface.co/t5-base tmp_correcter
mv tmp_correcter artifacts/correcter

echo "[5/6] Cloning LLM model..."
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2 tmp_llm
mv tmp_llm artifacts/llm

echo "[6/6] Preparing data and indexes..."
python3 main.py