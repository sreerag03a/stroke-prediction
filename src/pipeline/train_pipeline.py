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

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier


from src.model.model_trainer import evaluate_models



from imblearn.over_sampling import SMOTE,ADASYN,BorderlineSMOTE
from imblearn.pipeline import Pipeline

@dataclass
class ModelConfig:
    models_config = os.path.join('outputs','models')
    trained_model_path = os.path.join('outputs','models','trained_model.pkl')

class Train_Pipeline:

    def __init__(self):
        self.config = ModelConfig()

    def get_pipelines(self,models_dict:dict):
        oversamplers = {
            'SMOTE' : SMOTE(),
            'ADASYN': ADASYN(),
            'BorderlineSMOTE': BorderlineSMOTE()
        }

        pipelines = {}
        for oversampler_name,oversampler in oversamplers.items():
            for model_name,model in models_dict.items():
                pipelines[f'{oversampler_name}_{model_name}'] = Pipeline([
                    ('sampler', oversampler),
                    ('model',model)
                ])
        self.pipelines=pipelines
        return pipelines
    
    def train_models(self,train_set,test_set):
        os.makedirs(self.config.models_config, exist_ok=True)
        models = {
            "Logistic Regression" : LogisticRegression(),
            "Random Forest Classifier" : RandomForestClassifier(),
            "Support Vector Machine" : SVC(),
            "Gaussian Naive Bayes" : GaussianNB(),
            "Multinomial Naive Bayes" : MultinomialNB(),
            "XGBoost" : XGBClassifier()
        }
        params ={
            "Logistic Regression" : {},
            'Random Forest Classifier' : {},
            "Support Vector Machine" : {},
            "Gaussian Naive Bayes" : {},
            "Multinomial Naive Bayes" : {},
            "XGBoost" : {}
        }
        try:
            logging.info("Initializing train and test sets")
            X_train,y_train,X_test,y_test = (
                    train_set[:,:-1],
                    train_set[:,-1],
                    test_set[:,:-1],
                    test_set[:,-1]
                )
            models_path = os.path.join(self.config.models_config)
            pipelines = self.get_pipelines(models)
            results,fitted_models = evaluate_models(X_train,y_train,X_test,y_test,pipelines,params)
            finalized_models = {}
            for model, os_scores in results.items():
                max_score_index = np.argmax(list(os_scores.values()))
                print(f'{model} scores : {(os_scores.values())}')
                best_model = list(fitted_models[model].values())[max_score_index]
                finalized_models[model] = best_model
                print(f'{model} : {list(os_scores.values())[max_score_index]}')

            for modelname,model in finalized_models.items():
                savepath = os.path.join(models_path,f'{modelname}.pkl')
                save_obj(savepath,model)
                
            return 'finished'
                
        except Exception as e:
            raise CustomException(e,sys)

        
