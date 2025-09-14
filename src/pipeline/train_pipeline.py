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


from imblearn.over_sampling import SMOTE,ADASYN,BorderlineSMOTE,SMOTENC

@dataclass
class DataHandlingConfig:
    transformed_data_path:str = os.path.join('outputs','ogdata_oversampled.csv')

class ImbalanceHandling:
    def __init__(self):
        self.datahandleconfig = DataHandlingConfig()
    
    def start_oversample(self,train_data,oversampler):
        x_train = train_data[:,:-1]
        y_train = train_data[:,-1]

        logging.info("Started resample")
        x_resampled,y_resampled = oversampler.fit_resample(x_train,y_train)
        logging.info('Resample completed with new training set of {} rows'.format(x_resampled.shape[0]))

        # This was done because the resampling seemed to give values in between 0 and 1 for certain columns
        encode_columns = [1,2,3,4,5,6,7,9,10]
        x_resampled[:,encode_columns] = np.round(x_resampled[:,encode_columns])

        combined = np.c_[x_resampled,y_resampled]
        newdf = pd.DataFrame(combined)
        newdf.to_csv(self.datahandleconfig.transformed_data_path,index=False,header=False)
        return(
            x_resampled,
            y_resampled
        )


class Train_Pipeline:

    def __init__(self):
        pass

    def oversample_comb(self,train_data):
        oversamplers = {
            'SMOTE' : SMOTE(),
            'SMOTENC': SMOTENC(),
            'ADASYN': ADASYN(),
            'BorderlineSMOTE': BorderlineSMOTE()
        }

