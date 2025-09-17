import sys
import pandas as pd

from src.handling.exceptions import CustomException
from src.handling.logger import logging
from src.handling.utils import load_obj



class PredictPipeline:

    def __init__(self):
        pass


    def predictstroke(self,features,selected_model=None):
        try:
            pass
        except Exception as e:
            raise CustomException(e,sys)
