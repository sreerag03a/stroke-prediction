#basic utils
from src.components.handling import logger
import logging
from src.components.handling.exceptions import CustomException
import os
import sys
from src.components.handling.utils import save_obj
import src.components.model.model_trainer
from dataclasses import dataclass

#important modules
import pandas as pd
import numpy as np

#sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,FunctionTransformer,StandardScaler

datapath='data/healthcare-dataset-stroke-data.csv'

@dataclass
class DataConfig:
    train_data_path:str = os.path.join('outputs','train.csv')
    test_data_path:str = os.path.join('outputs','test.csv')
    original_data_path:str = os.path.join('outputs','ogdata.csv')

class DataIngestion:
    def __init__(self,split_ratio): 
        '''
        Split ratio - put value between 0 and 1 - This will be the ratio the training and testing data will be split into
        If split ratio is given as 0.3 - The data will be split 70:30 into training and testing sets
        
        '''
        self.ingest_conf = DataConfig()
        self.split_ratio = split_ratio

    def start_ingest(self):
        '''

        Data is loaded and split into training and testing sets and saved into data/

        '''
        
        logging.info('Data Ingestion started.')
        try:
            df = pd.read_csv(datapath)
            logging.info("Successfully loaded data as Dataframe")

            os.makedirs(os.path.dirname(self.ingest_conf.train_data_path), exist_ok= True)
            nrow,ncol = df.shape

            df.to_csv(self.ingest_conf.original_data_path, index=False,header=True)
            logging.info("Train test split initiated - Splitting {} datapoints into {} training points and {} test points".format(nrow,round((1-self.split_ratio)*nrow),round(self.split_ratio*nrow)))

            train_index, test_index = train_test_split(np.arange(nrow), test_size=self.split_ratio,random_state=76)
            train_set,test_set = df.iloc[train_index,:],df.iloc[test_index,:]

            train_set.to_csv(self.ingest_conf.train_data_path, index=False,header=True)
            test_set.to_csv(self.ingest_conf.test_data_path, index=False,header=True)

            return (
                self.ingest_conf.train_data_path,
                self.ingest_conf.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

@dataclass
class DataTransformConfig:
    preprocessor_obj_path = os.path.join('outputs','models','preprocessor.pkl')

class DataTransform:

    '''
    This is where the data is preprocessed. Here, the features are imputed, scaled and encoded.
    
    '''

    def __init__(self):
        self.data_transform_config = DataTransformConfig()

    
    def get_transformer_obj(self):
        try:
            

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler() )
                ]
            )      
            prerprocessor = ColumnTransformer(
                [
                ("OHencoder",OneHotEncoder(), ['gender']),
                ("LabelEncoder", OrdinalEncoder(),['smoking_status'])
                ]
            )
            logging.info("Numerical values Imputed and Standard Scaled")

            return prerprocessor
        except Exception as e:
            raise CustomException(e,sys)