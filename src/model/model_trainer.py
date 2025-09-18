from src.handling import logger
import logging
from src.handling.exceptions import CustomException
import os
import sys
from collections import Counter

from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,precision_recall_fscore_support,matthews_corrcoef,precision_recall_curve

from sklearn.metrics import precision_score,recall_score

    
def evaluate_models(X_train,y_train,X_test,y_test,pipelines,params):
#To tune different models based on evaluation metric
    try:
        results = {}
        fitted_models = {}
        all_scores= {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for i in range(len(list(pipelines.items()))):
            name,pipe = list(pipelines.items())[i]
            modelname = name.split('_')[1]
            samplername = name.split('_')[0]
            paramgrid = params[modelname]
            logging.info(f'Optimizing {modelname} with oversampler : {samplername}')
            gs = GridSearchCV(pipe,paramgrid,scoring = 'average_precision',cv=3,n_jobs = -2)
            gs.fit(X_train,y_train)

            fitted_model = gs.best_estimator_.named_steps['model']
            sampler = gs.best_estimator_.named_steps["sampler"]
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            logging.info(f"{name} oversampling -> before: {Counter(y_train)}, after: {Counter(y_res)}")


            threshold_val,y_proba = find_threshold(fitted_model,X_test,y_test)

            accuracy = fitted_model.score(X_test,y_test)

            roc_auc = roc_auc_score(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)


            '''
            This part is done to independently set the thresholds (to improve the )
            
            '''
            ypreds = (y_proba >= threshold_val).astype(int)

            precision,recall,fbeta,_ = precision_recall_fscore_support(y_test,ypreds,beta = 0.5)
            mcc = matthews_corrcoef(y_test,ypreds)
            f1score= f1_score(y_test,ypreds)
            scores = {'Accuracy': accuracy,
                      'F1 score' : f1score,
                      'F beta (0.5)': fbeta[1],
                      'Precision': precision[1],
                      'Recall': recall[1],
                      'ROC AUC': roc_auc,
                      'PR_AUC' : pr_auc,
                      'MCC': mcc}
            if modelname not in results:
                results[modelname] = {}
            if modelname not in fitted_models:
                fitted_models[modelname] = {}
            if modelname not in all_scores:
                all_scores[modelname] = {}
            results[modelname][samplername] = precision[1]
            fitted_models[modelname][samplername] = [fitted_model,threshold_val]
            all_scores[modelname][samplername] = scores
            
        return results,fitted_models,all_scores

    
    except Exception as e:
        raise CustomException(e,sys)
        


def find_threshold(model,X_test,y_test):

    '''
    To find the best threshold (at which the model classifies) so that it gives the best performance

    Here I have used a combination of recall and precision to maximize both.
    But from testing, the precision and accuracy takes a hit but this method seems to maximize recall, which is important for medical cases.
    
    '''
    
    y_proba = model.predict_proba(X_test)[:,1]

    precisions,recalls,thresholds = precision_recall_curve(y_test,y_proba)

    scores = []
    for t in thresholds:
        ypred = (y_proba >= t).astype(int)
        f1score = f1_score(y_test,ypred)
        comb_score = f1score
        scores.append(comb_score)

    best_index = scores.index(max(scores))
    return thresholds[best_index],y_proba


        
