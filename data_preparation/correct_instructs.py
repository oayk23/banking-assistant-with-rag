import pandas as pd
from banking_assistant.correcter import Correcter
from tqdm import tqdm
import os
from pathlib import Path

def correct(texts, correcter:Correcter):
    corrected_text = correcter.correct(texts=texts)
    return corrected_text

def batchify(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def correcting_pipeline(df_path:Path,artifacts_path:Path,save_path:Path,batch_size:int = 128):
    dataframe = pd.read_csv(df_path)
    correcter = Correcter(os.path.join(artifacts_path,"correcter"))
    texts = dataframe['instruction'].to_list()
    corrected_texts = []
    for batch_text in tqdm(batchify(texts,batch_size)):
        corrected_text = correct(batch_text,correcter)
        corrected_texts.extend(corrected_text)

    assert len(corrected_texts) == len(texts)

    dataframe['instruction_corrected'] = corrected_texts

    dataframe.to_csv(save_path,index=False)
