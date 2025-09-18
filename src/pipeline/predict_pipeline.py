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


    def predictstroke(self,features,selected_model):
        '''
        The main prediction function used in the Flask app to predict the occurence of stroke
        
        '''
        try:
            stacking,stacking_threshold = load_obj('outputs/models/Stacking Classifier.pkl')
            lsvc,lsvc_threshold = load_obj('outputs/models/Linear SVC.pkl')
            voting,voting_threshold = load_obj('outputs/models/Voting Classifier.pkl')

            preprocessor_path = 'outputs/models/preprocessor.pkl'
            logging.info('Loading preprocessor for data...')

            preprocessor_obj = load_obj(preprocessor_path)[0]
            
            logging.info('Attempting preprocessor')
            data_transformed = preprocessor_obj.transform(features)

            logging.info('Completed preprocessing - {}'.format(data_transformed))

            if selected_model == 'Majority Voting':

                logging.info('Loading model predictions')
                stacking_proba = stacking.predict_proba(data_transformed)[:,1]
                stacking_pred = (stacking_proba>= stacking_threshold).astype(int)

                lsvc_proba = lsvc.predict_proba(data_transformed)[:,1]
                lsvc_pred = (lsvc_proba>= lsvc_threshold).astype(int)

                voting_proba = voting.predict_proba(data_transformed)[:,1]
                voting_pred = (voting_proba>= voting_threshold).astype(int)
                probs = [stacking_proba,lsvc_proba,voting_proba]
                predictions = [stacking_pred[0],lsvc_pred[0],voting_pred[0]]
                logging.info(predictions)
                return maj_vote(predictions),(np.mean(probs))*100
            
            elif selected_model == 'Averaged Model':

                logging.info('Loading model predictions')
                stacking_proba = stacking.predict_proba(data_transformed)[:,1]

                lsvc_proba = lsvc.predict_proba(data_transformed)[:,1]

                voting_proba = voting.predict_proba(data_transformed)[:,1]

                probs = [stacking_proba,lsvc_proba,voting_proba]
                logging.info(probs)
                
                return averaging_prob(probs)
            else:
                modelpath = f'outputs/models/{selected_model}.pkl'
                model,threshold = load_obj(modelpath)
                model_proba = model.predict_proba(data_transformed)[:,1]

                model_pred = (model_proba >= threshold).astype(int)
                logging.info(f'Model prediction : {model_pred}')
                return model_pred,model_proba*100



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
        return (avg_prob > threshold).astype(int),avg_prob*100
    except Exception as e:
        raise CustomException(e,sys)



class CustomData:
    def __init__(self,datavector):
        self.datavector = datavector

    def get_dataframe(self):
        try:
            data_dict = {
                'gender': [self.datavector['gender']],
                'age':[int(self.datavector['age'])],
                'hypertension': [int(self.datavector['hypertension'])],
                'heart_disease': [int(self.datavector['heart_disease'])],
                'ever_married': [self.datavector['ever_married']],
                'work_type' : [self.datavector['work_type']],
                'Residence_type': [self.datavector['Residence_type']],
                'avg_glucose_level': [self.datavector['avg_glucose_level']],
                'bmi': [self.datavector['bmi']],
                'smoking_status': [self.datavector['smoking_status']]
            }
            return pd.DataFrame(data_dict)
        

        except Exception as e:
            raise CustomException(e,sys)