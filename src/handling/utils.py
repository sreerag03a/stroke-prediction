import os
import sys
import numpy as np
import pandas as pd
import dill
from src.handling.logger import logging
from src.handling.exceptions import CustomException
from kaggle.api.kaggle_api_extended import KaggleApi


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
        
def save_obj(filepath,obj):
    # To pickle models or fitted preprocessor

    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)

        with open(filepath,"wb") as fileobj:
            dill.dump(obj, fileobj)

    except Exception as e:
        raise CustomException(e,sys)

def load_obj(filepath):
    #Load pickled object
    try:
        with open(filepath, "rb") as f:
            return dill.load(f)
    except Exception as e:
        raise CustomException(e,sys)