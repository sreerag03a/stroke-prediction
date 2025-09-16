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
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier


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
            'ADASYN': ADASYN()
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
            "Logistic Regression" : LogisticRegression(class_weight="balanced"),
            "Random Forest Classifier" : RandomForestClassifier(),
            "Linear SVC" : LinearSVC(class_weight="balanced"),
            "XGBoost Classifier" : XGBClassifier(scale_pos_weight=21.5),
            "Gradient Boosting Classifier" : GradientBoostingClassifier(),
            "LightGBM Classifier" : LGBMClassifier(class_weight="balanced"),
            "K Neighbors Classifier": KNeighborsClassifier()

        }
        params ={
            "Logistic Regression" : {
                'model__max_iter':[1500,4000],
                'model__C': [0.001,0.01, 0.1, 1, 10],
                'model__penalty': ['l2'],
                'model__solver': ['lbfgs', 'liblinear', 'saga','sag']
            },
            'Random Forest Classifier' : {
                'model__n_estimators': [50, 100, 300],
                'model__max_depth': [5, 10, 15,20],
                'model__min_samples_split': [2, 5, 8],
                'model__min_samples_leaf': [1, 2, 3],
                'model__max_features': ['sqrt', 'log2'],
                'model__class_weight': ['balanced', 'balanced_subsample']
            },
            "Linear SVC" :{
                'model__C': [0.01,0.1, 1, 10, 100],
                'model__penalty' : ['l1','l2'],
                'model__max_iter':[1500,4000],
                'model__class_weight': ['balanced'],
                'model__dual': [False]
            },
            "XGBoost Classifier" : {
                'model__n_estimators': [100, 300],
                'model__max_depth': [3, 5, 7],
                'model__learning_rate': [0.01, 0.1],
                'model__subsample': [0.8, 1.0],
                'model__colsample_bytree': [0.8, 1.0]
            },
            "Gradient Boosting Classifier" : {
                'model__n_estimators': [100,300],
                'model__learning_rate': [0.01,0.1],
                'model__max_depth': [3, 5],
                'model__min_samples_leaf': [1, 2, 4]
            },
            "LightGBM Classifier" : {
                'model__n_estimators': [150,300],
                'model__learning_rate': [0.01,0.1],
                "model__max_depth": [3, 5],
                "model__num_leaves": [31, 63],
                "model__class_weight": ['balanced'],
                "model__boosting_type": ['gbdt','dart']
            },
            'K Neighbors Classifier': {
                'model__n_neighbors':[5,8,12],
                'model__weights':['distance','uniform'],
                'model__algorithm':['ball_tree','kd_tree','brute'],
                'model__leaf_size':[30,50]
            }
             
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
            results,fitted_models,all_scores = evaluate_models(X_train,y_train,X_test,y_test,pipelines,params)
            finalized_models = {}
            for model, os_scores in results.items():
                max_score_index = np.argmax(list(os_scores.values()))

                best_model = list(fitted_models[model].values())[max_score_index]
                best_model_oversampler = list(all_scores[model])[max_score_index]
                model_scores = all_scores[model][best_model_oversampler]
                print(f'{model}: {model_scores}')
                finalized_models[model] = best_model

            for modelname,model in finalized_models.items():
                savepath = os.path.join(models_path,f'{modelname}.pkl')
                save_obj(savepath,model)
                
            return 'finished'
                
        except Exception as e:
            raise CustomException(e,sys)

        
