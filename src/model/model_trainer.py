from src.handling import logger
import logging
from src.handling.exceptions import CustomException
import os
import sys
from collections import Counter

from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.metrics import recall_score,f1_score,roc_auc_score,average_precision_score


    
def evaluate_models(X_train,y_train,X_test,y_test,pipelines,params):
#To tune different models based on eval_metric
    try:
        results = {}
        fitted_models = {}
        all_scores= {}
        # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for i in range(len(list(pipelines.items()))):
            name,pipe = list(pipelines.items())[i]
            modelname = name.split('_')[1]
            samplername = name.split('_')[0]
            paramgrid = params[modelname]
            logging.info(f'Optimizing {modelname} with oversampler : {samplername}')
            gs = GridSearchCV(pipe,paramgrid,scoring = 'f1',cv=3,n_jobs = -2)
            gs.fit(X_train,y_train)

            fitted_model = gs.best_estimator_.named_steps['model']
            sampler = gs.best_estimator_.named_steps["sampler"]
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            logging.info(f"{name} oversampling -> before: {Counter(y_train)}, after: {Counter(y_res)}")

            accuracy = fitted_model.score(X_test,y_test)

            ypreds = fitted_model.predict(X_test)

            if hasattr(fitted_model,"predict_proba"):
                y_proba = fitted_model.predict_proba(X_test)[:,1]
            elif hasattr(fitted_model,"decision_function"):
                y_proba = fitted_model.decision_function(X_test)
            else:
                y_proba = None
            roc_auc = roc_auc_score(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)

            recall = recall_score(y_test,ypreds)
            f1score= f1_score(y_test,ypreds)
            scores = {'accuracy': accuracy,
                      'f1_score' : f1score,
                      'recall': recall,
                      'ROC AUC': roc_auc,
                      'PR_AUC' : pr_auc}
            if modelname not in results:
                results[modelname] = {}
            if modelname not in fitted_models:
                fitted_models[modelname] = {}
            if modelname not in all_scores:
                all_scores[modelname] = {}
            results[modelname][samplername] = accuracy
            fitted_models[modelname][samplername] = fitted_model
            all_scores[modelname][samplername] = scores
            
        return results,fitted_models,all_scores

    
    except Exception as e:
        raise CustomException(e,sys)
        





        
