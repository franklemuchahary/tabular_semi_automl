#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.ensemble import RandomForestRegressor
import sys
from skopt import BayesSearchCV
from sklearn import metrics


# In[2]:


##### Linear Regression
def Linear_Model1(dataset, target):
    '''Training Linear Model with All Dataset and returning Model, R^2 and RMSE'''
    #dataset1 = dataset[data.columns.intersection(data_type[data_type.Type.isin(['Numeric'])].ColumName)]
    dataset1 = dataset
    x = dataset1.drop(target, axis=1)
    y = dataset1[target]
    linreg = LinearRegression()
    linreg.fit(x,y)
    r_Square = linreg.score(x,y)
    RMSE = np.sqrt(mean_squared_error(y, linreg.predict(x)))
    MAE = mean_absolute_error(y, linreg.predict(x))
    predicted_vs_actuals = pd.DataFrame({
        'Predicted': y,
        'Actuals': linreg.predict(x)
    })
    
    model_coefs = pd.DataFrame({
        'Columns': ['Intercept'] + list(x.columns.values),
        'Coefficients': [linreg.intercept_]+list(linreg.coef_.flatten())
    })
    
    return linreg, r_Square, RMSE, MAE, predicted_vs_actuals, model_coefs


# In[3]:


def ElasticNet_Model1(dataset, target, cv=5):
    '''Training ElasticNetCV Model with All Dataset and returning Model, R^2 and RMSE'''
    #dataset1 = dataset[data.columns.intersection(data_type[data_type.Type.isin(['Numeric'])].ColumName)]
    
    dataset1 = dataset
    x = dataset1.drop(target, axis=1)
    y = dataset1[target]
    ElasticNet_Reg = ElasticNetCV(cv= cv, verbose=10)
    ElasticNet_Reg.fit(x,y)
    r_Square = ElasticNet_Reg.score(x,y)
    RMSE = np.sqrt(mean_squared_error(y, ElasticNet_Reg.predict(x)))
    MAP = mean_absolute_error(y, ElasticNet_Reg.predict(x))
    
    predicted_vs_actuals = pd.DataFrame({
        'Predicted': y,
        'Actuals': ElasticNet_Reg.predict(x)
    })
    
    alpha_vs_mse = pd.DataFrame({
        'Alpha': np.mean(ElasticNet_Reg.mse_path_, axis=1),
        'MSE': ElasticNet_Reg.alphas_
    })
    
    model_coefs = pd.DataFrame({
        'Columns': ['Intercept'] + list(x.columns.values),
        'Coefficients': [ElasticNet_Reg.intercept_]+list(ElasticNet_Reg.coef_.flatten())
    })
    
    return ElasticNet_Reg, r_Square, RMSE, MAP, predicted_vs_actuals, model_coefs




class CrossValModelBuilderRegression():
    def __init__(self, model_object="", n_folds=5, target_variable=''):
        self.model = model_object
        self.n_folds = n_folds
        self.kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
    def _get_train_val_data(self, X_train, Y, train_idx, val_idx):
        t_x = X_train.iloc[train_idx]
        v_x = X_train.iloc[val_idx]
        t_y = Y[train_idx]
        v_y = Y[val_idx]
        
        return(t_x, v_x, t_y, v_y)
    
    def fit_base_cv_model(self, X_train, Y, model_logging_file=""):
        strf_split = self.kf
        open(model_logging_file, 'w').close()
        
            
        validation_scores = []
        models_list = []
        cv_counter=1
        all_predictions = np.zeros(len(X_train))
        
        for train_idx, val_idx in strf_split.split(X_train, Y):
            sys.stdout=open(model_logging_file,"a")
    
            print("====== Validation Fold Number: ", cv_counter, " ======", end="\n\n")
    
            t_x, v_x, t_y, v_y = self._get_train_val_data(X_train, Y, train_idx, val_idx)   
    
            model = self.model
            model.fit(t_x, t_y)
    
            models_list.append(model)
    
            
            r_Square = model.score(t_x, t_y)
            RMSE = np.sqrt(metrics.mean_squared_error(v_y, model.predict(v_x)))
            MAP = metrics.mean_absolute_error(v_y, model.predict(v_x))
            val_preds = model.predict(v_x)

            print("\n\n")
            print("Validation RMSE : ", RMSE, end="\n\n")


            print("\n\n")
            print("Validation MAP: ", MAP, end="\n\n")
                
            validation_scores.append(MAP)
            all_predictions[val_idx] += val_preds/self.n_folds
            
    
            cv_counter+=1
    
            print("============"*8, end="\n\n\n\n")
            sys.stdout.flush()
            
        predicted_vs_actuals = pd.DataFrame({
                'Predicted': all_predictions,
                'Actuals': Y
            })
            
        return (models_list, predicted_vs_actuals)
    
    
    def fit_bayesian_tuning_model(self, X_train, Y, search_spaces="", parameter_tuning_logging_file="", 
                           write_tuning_results_path=""):  
        
        scoring = 'r2'
        
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
            
            scoring_metric = "R-Squared"

    
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
        
        tuned_model_predictions = tuned_model.predict(X_train)
        predicted_vs_actuals = pd.DataFrame({
                'Predicted': tuned_model_predictions,
                'Actuals': Y
            })
        
        
        return (tuned_model, predicted_vs_actuals)


    
def score_regression_models(predicted_vs_actuals_df):
    r2 = metrics.r2_score(predicted_vs_actuals_df['Actuals'], predicted_vs_actuals_df['Predicted'])
    mae = metrics.mean_absolute_error(predicted_vs_actuals_df['Actuals'], predicted_vs_actuals_df['Predicted'])
    rmse = metrics.mean_absolute_error(predicted_vs_actuals_df['Actuals'], predicted_vs_actuals_df['Predicted'])
    
    return(r2, mae, rmse)    