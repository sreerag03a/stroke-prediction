import os
import sys
import numpy as np

from src.handling.logger import logging
from src.handling.exceptions import CustomException  
from src.handling.utils import download_data
from src.model.data_transform import DataIngestion,DataTransform
from src.pipeline.train_pipeline import Train_Pipeline

DATA_DIR = os.path.join(os.getcwd(),"data")
os.makedirs(DATA_DIR,exist_ok=True)


from imblearn.over_sampling import SMOTE,ADASYN,BorderlineSMOTE

if __name__ == "__main__":
    try:
        logging.info("Attempting to download stroke dataset from kaggle...")
        download_data(DATA_DIR)

        print("Check outputs/logs for logs if something goes wrong")
        logging.info('Downloaded stroke dataset.')

        data_ingest = DataIngestion(0.3)
        train_path,test_path = data_ingest.start_ingest()
        data_transform = DataTransform()
        incomplete_train_data,test_data = data_transform.start_transform(train_path,test_path)

        # print(np.where(np.isnan(incomplete_train_data)))
        trainer = Train_Pipeline()
        models = trainer.train_models(incomplete_train_data,test_data)
        print(models)
    except Exception as e:
        raise CustomException(e,sys)