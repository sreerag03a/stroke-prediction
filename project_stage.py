import os
import sys
import numpy as np

from src.components.handling.logger import logging
from src.components.handling.exceptions import CustomException  
from src.components.handling.utils import download_data


DATA_DIR = os.path.join(os.getcwd(),"data")
os.makedirs(DATA_DIR,exist_ok=True)




if __name__ == "__main__":
    try:
        download_data(DATA_DIR)
    except Exception as e:
        raise CustomException(e,sys)