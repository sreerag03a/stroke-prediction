from src.handling import logger
import logging
from src.handling.exceptions import CustomException
import os
import sys

from sklearn.model_selection import GridSearchCV,StratifiedKFold



    
def evaluate_models(X_train,y_train,X_test,y_test,pipelines,params):
#To tune different models based on eval_metric
    try:
        results = {}
        fitted_models = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for i in range(len(list(pipelines.items()))):
            name,pipe = list(pipelines.items())[i]
            modelname = name.split('_')[1]
            samplername = name.split('_')[0]
            paramgrid = params[modelname]
            logging.info(f'Optimizing {modelname} with oversampler : {samplername}')
            gs = GridSearchCV(pipe,paramgrid,scoring = 'f1',cv=cv,n_jobs = -1)
            gs.fit(X_train,y_train)

            fitted_model = gs.best_estimator_.named_steps['model']
            test_score = fitted_model.score(X_test,y_test)

            if modelname not in results:
                results[modelname] = {}
            if modelname not in fitted_models:
                fitted_models[modelname] = {}
            results[modelname][samplername] = test_score
            fitted_models[modelname][samplername] = fitted_model
        return results,fitted_models

    
    except Exception as e:
        raise CustomException(e,sys)
        





        
