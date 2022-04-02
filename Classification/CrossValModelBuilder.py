import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, log_loss
from skopt import BayesSearchCV


class CrossValModelBuilder():
    def __init__(self, model_object="", n_folds=5, target_variable=''):
        self.model = model_object
        self.n_folds = n_folds
        self.kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        if target_variable.nunique() > 2:
            self.is_binary = False
        else:
            self.is_binary = True
        
    def _get_train_val_data(self, X_train, Y, train_idx, val_idx):
        t_x = X_train.iloc[train_idx]
        v_x = X_train.iloc[val_idx]
        t_y = Y[train_idx]
        v_y = Y[val_idx]
        
        return(t_x, v_x, t_y, v_y)
    
    def fit_base_cv_model(self, X_train, Y, model_logging_file=""):
        strf_split = self.kf
        open(model_logging_file, 'w').close()
        
        if self.is_binary:
            all_predictions = np.zeros(len(X_train))
        else:
            all_predictions = np.zeros((len(X_train), Y.nunique()))
            
        validation_scores = []
        models_list = []
        cv_counter=1
        
        for train_idx, val_idx in strf_split.split(X_train, Y):
            sys.stdout=open(model_logging_file,"a")
    
            print("====== Validation Fold Number: ", cv_counter, " ======", end="\n\n")
    
            t_x, v_x, t_y, v_y = self._get_train_val_data(X_train, Y, train_idx, val_idx)   
    
            model = self.model
            model.fit(t_x, t_y)
    
            models_list.append(model)
    
            if self.is_binary == True:
                val_preds = model.predict_proba(v_x)[:, 1]
                val_score = roc_auc_score(v_y, val_preds)
                all_predictions[val_idx] += val_preds/self.n_folds  
                
                print("\n\n")
                print("Validation Current Fold AUC Score: ", val_score, end="\n\n")
            else:
                val_preds = model.predict_proba(v_x)
                val_score = log_loss(v_y, val_preds)
                all_predictions[val_idx] += val_preds/self.n_folds  
                
                print("\n\n")
                print("Validation Current Fold LogLoss: ", val_score, end="\n\n")
                
            validation_scores.append(val_score)
            
    
            cv_counter+=1
    
            print("============"*8, end="\n\n\n\n")
            sys.stdout.flush()
        
        
        if self.is_binary:
             predicted_vs_actuals = pd.DataFrame({
                'Predicted': all_predictions,
                'Actuals': Y
            })
        else:
            predicted_vs_actuals = pd.DataFrame(all_predictions)
            predicted_vs_actuals.columns = models_list[0].classes_   
            predicted_vs_actuals['Actuals'] = Y
            
        return (models_list, predicted_vs_actuals)
    
    
    def fit_bayesian_tuning_model(self, X_train, Y, search_spaces="", parameter_tuning_logging_file="", 
                           write_tuning_results_path=""):
        
        if self.is_binary:
            scoring = 'roc_auc'
        else:
            scoring = 'neg_log_loss'
        
        bayes_cv_tuner = BayesSearchCV(
            estimator = self.model,
            search_spaces = search_spaces,    
            scoring = scoring,
            cv = self.kf,
            n_jobs = -1,
            n_points = 3,
            n_iter = 10,   
            verbose = 0,
            refit = True,
            random_state = 42
        )
        
        def status_print_bayesian_tuning(optim_result, 
                                          parameter_tuning_logging_file=parameter_tuning_logging_file,
                                         write_file_base_path=write_tuning_results_path):
            """Status callback durring bayesian hyperparameter search"""
            #Get all the models tested so far in DataFrame format
            all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
    
            #Get current parameters and the best parameters    
            best_params = pd.Series(bayes_cv_tuner.best_params_)
    
            sys.stdout=open(parameter_tuning_logging_file,"a")
            
            if self.is_binary:
                scoring_metric = "AUC"
            else:
                scoring_metric = "LogLoss"
    
            print('Model #{}\nBest {} : {}\nBest params: {}\n'.format(
                len(all_models),
                scoring_metric,
                np.round(np.abs(bayes_cv_tuner.best_score_), 4),
                bayes_cv_tuner.best_params_
            ))
    
            sys.stdout.flush()
    
            #Save all model results
            clf_name = bayes_cv_tuner.estimator.__class__.__name__
            all_models.to_csv(write_file_base_path+clf_name+"_cv_results.csv", index=False)
        
        
        
        open(parameter_tuning_logging_file, 'w').close()
        result = bayes_cv_tuner.fit(X_train.values, Y.values, 
                    callback=status_print_bayesian_tuning)
        
        self.model = self.model.set_params(**result.best_params_)
        tuned_model = self.model.fit(X_train, Y)
        
        if self.is_binary:
            tuned_model_predictions = tuned_model.predict_proba(X_train)[:, 1]
            predicted_vs_actuals = pd.DataFrame({
                'Predicted': tuned_model_predictions,
                'Actuals': Y
            })
        else:
            tuned_model_predictions = tuned_model.predict_proba(X_train)
            predicted_vs_actuals = pd.DataFrame(tuned_model_predictions)
            predicted_vs_actuals.columns = tuned_model.classes_
            predicted_vs_actuals['Actuals'] = Y
        
        return (tuned_model, predicted_vs_actuals)