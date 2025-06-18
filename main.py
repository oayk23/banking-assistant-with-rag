from data_preparation import correcting_pipeline
from index_preparation import IndexHandler
from banking_assistant.classifier.train import build_models,train_and_select_best

import os
import pandas as pd
from pathlib import Path

def main():
    data_path = Path("data")
    dataframe_path = Path(os.path.join(data_path,"dataframe.csv"))
    corrected_save_path = Path(os.path.join(data_path,"corrected_dataframe.csv"))
    artifacts_path = Path("artifacts")
    correcting_pipeline(artifacts_path=artifacts_path,df_path=dataframe_path,save_path=corrected_save_path)

    index_handler = IndexHandler(Path(os.path.join(artifacts_path,"embedder")),dataframe_path=dataframe_path)
    index_handler.create_indexes("indexes")

    print("training classifiers...")

    dataframe = pd.read_csv(corrected_save_path)
    x = dataframe['instruction_corrected']
    y = dataframe['intent']
    best_name,best_model = train_and_select_best(x,y,save_path=os.path.join(artifacts_path,"classifier","classifier.pkl"))
    print("All OK!")

if __name__ == "__main__":
    main()