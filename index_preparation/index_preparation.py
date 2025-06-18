from banking_assistant.embedder import Embedder

from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import numpy as np
import faiss
import os
from pathlib import Path
import pickle

class IndexHandler:
    def __init__(self,embedder_path,dataframe_path):
        self.embedder = Embedder(embedder_path)
        self.dataframe = pd.read_csv(dataframe_path)


    def get_embeddings(self,texts,batch_size=128):
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            embeddings.append(self.embedder.embed(batch).cpu().numpy())

        return np.vstack(embeddings)
    
    @staticmethod
    def load_indexes_and_docs(indexes_path,intent):
        index = faiss.read_index(str(os.path.join(indexes_path,intent,"index.index")))
        with open(os.path.join(indexes_path,intent,"docs.pkl"),"rb") as docs_fp:
            docs = pickle.load(docs_fp)
        
        return index,docs
    
    def create_indexes(self,indexes_path):
        response_embeddings = self.get_embeddings(texts=self.dataframe['response'].to_list(),batch_size=128)
        category_to_embeddings = defaultdict(list)
        category_to_texts = defaultdict(list)
        categories = self.dataframe['intent'].to_list()
        sentences = self.dataframe['response'].to_list()

        for emb, cat, text in zip(response_embeddings, categories, sentences):
            category_to_embeddings[cat].append(emb)
            category_to_texts[cat].append(text)
        
        category_to_faiss = {}
        category_to_id_map = {}

        for cat, embs in category_to_embeddings.items():
            emb_array = np.array(embs).astype('float32')
            
            faiss.normalize_L2(emb_array)
            index = faiss.IndexFlatIP(emb_array.shape[1])
            index.add(emb_array) # type: ignore

            category_to_faiss[cat] = index
            category_to_id_map[cat] = category_to_texts[cat]

        os.makedirs(indexes_path,exist_ok=True)
        for intent,index in category_to_faiss.items():
            intent_path = os.path.join(indexes_path,intent)
            os.makedirs(intent_path)
            faiss.write_index(index,os.path.join(intent_path,"index.index"))
        for intent,docs in category_to_id_map.items():
            intent_path = os.path.join(indexes_path,intent)
            os.makedirs(intent_path,exist_ok=True)
            with open(os.path.join(intent_path,"docs.pkl"),"wb") as fp:
                pickle.dump(docs,fp)
        print("indexes and docs saved.")
