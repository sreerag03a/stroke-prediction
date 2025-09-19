import os
import sys
import numpy as np
import pandas as pd
import dill
from src.handling.logger import logging
from src.handling.exceptions import CustomException
from kaggle.api.kaggle_api_extended import KaggleApi
import json
import matplotlib.pyplot as plt




def download_data(data_dir):
    data_dict = {'healthcare-stroke.csv' : "fedesoriano/stroke-prediction-dataset"}
    for filename,link in data_dict.items():
        save_path = os.path.join(data_dir,filename)
        logging.info(f"Downloading {filename}")
        try:
            api = KaggleApi()
            api.authenticate()
            logging.info('api authentication complete')
            os.makedirs(data_dir, exist_ok=True)
            api.dataset_download_files(link, path=data_dir, unzip=True)
            

            logging.info("Dataset download completed.")
            src_file = os.path.join(data_dir, "healthcare-dataset-stroke-data.csv")
            logging.info(f'Dataset successfully saved at {src_file}')
        except Exception as e:
            raise CustomException(e,sys)
        
def save_obj(filepath,obj,*args):
    # To pickle models or fitted preprocessor

    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)

        with open(filepath,"wb") as fileobj:
            dill.dump((obj,*args), fileobj)

    except Exception as e:
        raise CustomException(e,sys)
    

def load_obj(filepath):
    #Load pickled object
    try:
        with open(filepath, "rb") as f:
            return dill.load(f)
    except Exception as e:
        raise CustomException(e,sys)
    

def metrics_img():
    with open("outputs/model_scores.json", "r") as f:
        scores = json.load(f)


    df = pd.DataFrame(scores).T.round(4)


    fig, ax = plt.subplots(figsize=(15, 4))
    ax.axis("off")
    df.index = df.index.str.replace(r'_.+$', '', regex=True)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1)  

    # Saving model scores table as image
    plt.savefig("outputs/metrics.png", dpi=300, bbox_inches="tight")

