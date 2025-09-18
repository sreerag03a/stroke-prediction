#basic utils
from src.handling import logger
import logging
from src.handling.exceptions import CustomException
import os
import sys
from src.handling.utils import save_obj
from dataclasses import dataclass
import json

#important modules
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier, VotingClassifier,StackingClassifier
from sklearn.svm import LinearSVC,SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.model.model_trainer import evaluate_models



from imblearn.over_sampling import SMOTE,ADASYN,BorderlineSMOTE
from imblearn.combine import SMOTEENN,SMOTETomek
from imblearn.pipeline import Pipeline
from imblearn import FunctionSampler

@dataclass
class ModelConfig:
    models_config = os.path.join('outputs','models')
    trained_model_path = os.path.join('outputs','models','trained_model.pkl')
    model_scores_path = os.path.join('outputs','model_scores.json')

class Train_Pipeline:

    def __init__(self):
        self.config = ModelConfig()

    def get_pipelines(self,models_dict:dict):
        oversamplers = {
            'None': FunctionSampler(func=None), # No oversampling
            'SMOTE' : SMOTE(),
            'ADASYN': ADASYN(),
            'SMOTEENN':SMOTEENN()
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
        base_models = [
            ('lr', LogisticRegression(max_iter=10000,class_weight='balanced')),
            ('rf', RandomForestClassifier(class_weight='balanced')),
            ('svc', SVC(probability=True,class_weight='balanced'))
        ]
        models = {
            "Linear SVC" : CalibratedClassifierCV(LinearSVC(max_iter=10000),method = 'sigmoid',cv = 5),
            "XGBoost Classifier" : XGBClassifier(scale_pos_weight=21.5),
            "Gradient Boosting Classifier" : GradientBoostingClassifier(),
            "K Neighbors Classifier": KNeighborsClassifier(),
            "Voting Classifier" : VotingClassifier(estimators=base_models,voting='soft'),
            "Stacking Classifier": StackingClassifier(estimators=base_models,final_estimator=LGBMClassifier(class_weight='balanced'))

        }
        params ={
            "Linear SVC" :{
                'model__estimator__C': [0.01,0.1, 1, 10, 100],
                'model__estimator__penalty' : ['l1','l2'],
                'model__estimator__class_weight': ['balanced'],
                'model__estimator__dual': [False]
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
            'K Neighbors Classifier': {
                'model__n_neighbors':[5,8,12],
                'model__weights':['distance','uniform'],
                'model__algorithm':['ball_tree','kd_tree','brute'],
                'model__leaf_size':[30,50]
            },
            "Voting Classifier" : {
                # 'model__lr__C': [0.01, 0.1, 1, 10],
                'model__lr__solver': ['lbfgs', 'liblinear', 'saga'],
                'model__rf__n_estimators': [100, 300],
                # 'model__rf__max_depth': [5, 10],
                'model__svc__C': [0.1, 1, 10],
                # 'model__svc__kernel': ['linear', 'rbf']
            },
            "Stacking Classifier": {
                # 'model__lr__C': [0.01, 0.1, 1, 10],
                'model__lr__solver': ['lbfgs', 'liblinear', 'saga'],
                'model__rf__n_estimators': [150, 300],
                # 'model__rf__max_depth': [5, 10],
                'model__svc__C': [0.1, 1, 10],
                # 'model__svc__kernel': ['linear', 'rbf'],
                'model__final_estimator__n_estimators': [150,300],
                'model__final_estimator__learning_rate': [0.01,0.1],
                # "model__final_estimator__max_depth": [3, 5],
                # "model__final_estimator__num_leaves": [31, 63],
                # "model__final_estimator__min_child_samples":[40,80]
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
            saved_scores = {}
            for model, os_scores in results.items():
                max_score_index = np.argmax(list(os_scores.values()))

                best_model,threshold = list(fitted_models[model].values())[max_score_index]
                best_model_oversampler = list(all_scores[model])[max_score_index]
                model_scores = all_scores[model][best_model_oversampler]
                print(f'{model}: {model_scores}')
                saved_scores[model] = model_scores
                finalized_models[model] = [best_model,threshold]
            with open(self.config.model_scores_path,'w') as f:
                json.dump(saved_scores,f,indent=4)

            for modelname,model in finalized_models.items():
                savepath = os.path.join(models_path,f'{modelname}.pkl')
                save_obj(savepath,model[0],model[1])
                
            return 'finished'
                
        except Exception as e:
            raise CustomException(e,sys)

