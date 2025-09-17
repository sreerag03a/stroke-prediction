import sys
import pandas as pd
import numpy as np

from src.handling.exceptions import CustomException
from src.handling.logger import logging
from src.handling.utils import load_obj


from collections import Counter


class PredictPipeline:

    def __init__(self):
        pass


    def predictstroke(self,features,selected_model=None):
        '''
        The main prediction function used in the Flask app to predict the occurence of stroke
        
        '''
        try:
            lgbm,lgbm_threshold = load_obj('outputs/models/LightGBM Classifier.pkl')
            lsvc,lsvc_threshold = load_obj('outputs/models/Linear SVC.pkl')
            lgr, lgr_threshold = load_obj('outputs/models/Logistic Regression.pkl')

            preprocessor_path = 'outputs/models/preprocessor.pkl'
            logging.info('Loading preprocessor for data...')

            preprocessor_obj = load_obj(preprocessor_path)
            
            logging.info('Attempting preprocessor')
            data_transformed = preprocessor_obj.transform(features)

            logging.info('Completed preprocessing - {}'.format(data_transformed))

            if selected_model == 'Majority Voting':

                logging.info('Loading model predictions')
                lgbm_proba = lgbm.predict_proba(data_transformed)[:,1]
                lgbm_pred = (lgbm_proba>= lgbm_threshold).astype(int)

                lsvc_proba = lsvc.predict_proba(data_transformed)[:,1]
                lsvc_pred = (lsvc_proba>= lsvc_threshold).astype(int)

                lgr_proba = lgr.predict_proba(data_transformed)[:,1]
                lgr_pred = (lgr_proba>= lgr_threshold).astype(int)

                predictions = [lgbm_pred,lsvc_pred,lgr_pred]
                return maj_vote(predictions)
            
            elif selected_model == 'Averaged Model':

                logging.info('Loading model predictions')
                lgbm_proba = lgbm.predict_proba(data_transformed)[:,1]

                lsvc_proba = lsvc.predict_proba(data_transformed)[:,1]

                lgr_proba = lgr.predict_proba(data_transformed)[:,1]

                probs = [lgbm_proba,lsvc_proba,lgr_proba]
                
                return averaging_prob(probs)
            else:
                modelpath = f'outputs/models/{selected_model}.pkl'
                model,threshold = load_obj(modelpath)
                model_proba = model.predict_proba(data_transformed)

                model_pred = (model_proba >= threshold).astype(int)
                return model_pred



        except Exception as e:
            raise CustomException(e,sys)


def maj_vote(predictions):
    try:
        return Counter(predictions).most_common(1)[0][0]
    except Exception as e:
        raise CustomException(e,sys)

def averaging_prob(probs):
    try:
        threshold = 0.3 # Set from independent testing of ensemble classification
        avg_prob = np.mean(probs)
        return (avg_prob > threshold).astype(int)
    except Exception as e:
        raise CustomException(e,sys)



class CustomData:
    def __init__(self,datavector):
        self.datavector = datavector

    def get_dataframe(self):
        try:
            data_dict = {
                'gender': self.datavector['gender'],
                'age':self.datavector['age'],
                'hypertension': self.datavector['hypertension'],
                'heart_disease': self.datavector['heart_disease'],
                'ever_married': self.datavector['ever_married'],
                'work_type' : self.datavector['work_type'],
                'Residence_type': self.datavector['Residence_type'],
                'avg_glucose_level': self.datavector['avg_glucose_level'],
                'bmi': self.datavector['bmi'],
                'smoking_status': self.datavector['smoking_status']
            }
            return pd.DataFrame(data_dict)
        

        except Exception as e:
            raise CustomException(e,sys)