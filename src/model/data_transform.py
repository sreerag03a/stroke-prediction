#basic utils
from src.handling import logger
import logging
from src.handling.exceptions import CustomException
import os
import sys
from src.handling.utils import save_obj
from dataclasses import dataclass

#important modules
import pandas as pd
import numpy as np

#sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,FunctionTransformer,StandardScaler

datapath='data/healthcare-dataset-stroke-data.csv'

@dataclass
class DataConfig:
    train_data_path:str = os.path.join('outputs','train.csv')
    test_data_path:str = os.path.join('outputs','test.csv')
    original_data_path:str = os.path.join('outputs','ogdata.csv')

class DataIngestion:
    def __init__(self,split_ratio=0.3): 
        '''
        Split ratio - put value between 0 and 1 - This will be the ratio the training and testing data will be split into
        If split ratio is given as 0.3 - The data will be split 70:30 into training and testing sets
        
        '''
        self.ingest_conf = DataConfig()
        self.split_ratio = split_ratio
    
    def start_ingest(self):
        '''
        This is where the data is split into sets and then saved into different files
        '''
        logging.info("Data Ingestion started")
        try:
            df = pd.read_csv(datapath)
            
            df_copy = df.iloc[:,[1,2,3,4,8,9,10,11]]
            logging.info('Successfully ingested data as a Dataframe')
            for col in ["hypertension","heart_disease"]:
                df_copy[col] = df_copy[col].astype(int)
            os.makedirs(os.path.dirname(self.ingest_conf.train_data_path), exist_ok= True)
            nrow,ncol = df.shape

            df.to_csv(self.ingest_conf.original_data_path, index=False,header=True)
            logging.info("Train test split initiated - Splitting {} datapoints into {} training points and {} test points".format(nrow,round((1-self.split_ratio)*nrow),round(self.split_ratio*nrow)))
            
            train_index, test_index = train_test_split(np.arange(nrow), test_size=self.split_ratio,random_state=76)
            train_set,test_set = df_copy.iloc[train_index,:],df_copy.iloc[test_index,:]

            train_set.to_csv(self.ingest_conf.train_data_path, index=False,header=True)
            test_set.to_csv(self.ingest_conf.test_data_path, index=False,header=True)
            self.datacolumns = df_copy.columns
            return (
                self.ingest_conf.train_data_path,
                self.ingest_conf.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)


'''
Data transformation

'''
@dataclass
class DataTransformConfig:
    preprocessor_obj_path:str = os.path.join('outputs','models','preprocessor.pkl')
    

class DataTransform:

    '''
    This is where the data is preprocessed. Here, the features are imputed, scaled and encoded.
    
    '''

    def __init__(self):
        self.data_transform_config = DataTransformConfig()
    
    def get_transformer_obj(self):
        try:
            
            impute_column = ['bmi']

            encode_columns = ['gender','smoking_status']

            numerical_cols = ['avg_glucose_level','bmi']

            self.OHencoder = OneHotEncoder(dtype=int)

            preprocessor = ColumnTransformer(
                [('imputer',SimpleImputer(strategy='mean'),impute_column),
                ('encoder',self.OHencoder,encode_columns),
                
                ],remainder='passthrough'
            )   
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_transform(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data')
            logging.info('Obtaining preprocessing object')

            preprocessor_obj = self.get_transformer_obj()

            target_column = "stroke"

            train_df_mod = train_df.drop(columns=[target_column], axis = 1)
            target_train_df = train_df[target_column]

            test_df_mod = test_df.drop(columns=[target_column], axis = 1)
            target_test_df = test_df[target_column]

            logging.info("Preprocessing data...")

            train_arr = preprocessor_obj.fit_transform(train_df_mod)
            test_arr = preprocessor_obj.transform(test_df_mod)

            train_arr = np.c_[train_arr,np.array(target_train_df)]
            test_arr = np.c_[test_arr,np.array(target_test_df)]

            logging.info("Preprocess completed")

            save_obj(

                filepath = self.data_transform_config.preprocessor_obj_path,
                obj = preprocessor_obj
            )
            logging.info("Saved preprocessing object in outputs folder")
            return (
                train_arr,
                test_arr
            )

        except Exception as e:
            logging.info("Error occured in transformation/preprocessing")
            raise CustomException(e,sys)