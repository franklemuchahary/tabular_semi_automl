from flask import Flask, render_template, request
from flask_restful import Resource, Api, reqparse
import werkzeug
import os
import json
from urllib.parse import unquote
import pickle
import pandas as pd

from Classification.DetectColumnTypes import DetectColumnTypes
from Classification.BaselineDataPrep import BaselineDataPrep
from Classification.CrossValModelBuilder import CrossValModelBuilder
from Classification.ClassificatonModelEvaluation import ClassificatonModelEvaluation
from Regression.RegressionModels import ElasticNet_Model1
from Regression.RegressionModels import Linear_Model1

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.externals import joblib
from Regression.RegressionModels import CrossValModelBuilderRegression
from Regression.RegressionModels import score_regression_models
from FeatureEngineering.FeatureEngineering import get_feature_engieering_variables


app = Flask(__name__)
api = Api(app)

@app.route('/detect_column_types/')
def detect_column_types():
    try:
    
        json_data = json.loads(unquote(request.args.get('json')))
    
        data_path = json_data['data_path']
        project_path = json_data['project_path']
    
        write_dtypes_file_path = project_path+'my_dtypes_file.csv'
    
        my_dtypes = DetectColumnTypes().get_column_types(data_path, write_dtypes_file_path)
    
        return_json = {
            'column_data_types': my_dtypes.to_json(),
            'csv_filepath_column_dtypes': write_dtypes_file_path
        }
    
    except Exception as e:
        return_json = {
                'Status':'Error : ' + str(e), 
            }
    
    return json.dumps(return_json)


@app.route('/do_feature_engineering/')
def do_feature_engineering():
    try:
        json_data = json.loads(unquote(request.args.get('json')))
        
        data_path = json_data['data_path']                 
        data_dtypes_path = json_data['data_dtypes_path']
        target_variable = json_data['target_variable']
        project_path = json_data['project_path']
    
        data = DetectColumnTypes().read_data(data_path)
        my_dtypes = DetectColumnTypes().read_data(data_dtypes_path)
    
        base_data_prep = BaselineDataPrep(data, my_dtypes)
        data = base_data_prep.get_baseline_prepared_data(one_hot_encode=False, target=target_variable)
        final_feature_engineered_df, all_columns, number_of_features_engineered = get_feature_engieering_variables(data,
                                                                                                                  my_dtypes,
                                                                                                                 target_variable)
    
        final_feature_engineered_df.to_csv(project_path+'feature_engineered_df.csv', index=False)
        
        return_json = {
            'feature_engineered_df_path': project_path+'feature_engineered_df.csv',
            'number_of_new_features': number_of_features_engineered,
            'all_features_generated': all_columns
        }
    
    except Exception as e:
        return_json = {
                'Status':'Error : ' + str(e), 
            }
    
    return json.dumps(return_json)


#####################################################################################

@app.route('/baseline_model_train_rf/')
def baseline_model_train_rf():
    try:
        json_data = json.loads(unquote(request.args.get('json')))
    
        data_path = json_data['data_path']                 
        data_dtypes_path = json_data['data_dtypes_path']
        target_variable = json_data['target_variable']
        project_path = json_data['project_path']
    
        data = DetectColumnTypes().read_data(data_path)
        my_dtypes = pd.read_csv(data_dtypes_path)
    
        base_data_prep = BaselineDataPrep(data, my_dtypes)
        data = base_data_prep.get_baseline_prepared_data(one_hot_encode=True, target=target_variable)
    
        X_train = data[data.columns.difference([target_variable])]
        Y = data[target_variable]
    
        params = {
            'verbose':50,
            'bootstrap': True,
            'n_estimators': 30, 
            'random_state':42
        }
        rf = RandomForestClassifier(**params)
        cval_model_builder = CrossValModelBuilder(model_object=rf, target_variable=Y)
        models_list, model_all_preds = cval_model_builder.fit_base_cv_model(X_train, Y, 
                                                    project_path+'baseline_model_train_rf_log.txt')
    
        joblib.dump(models_list[0], project_path+'baseline_rf_model.pk')
    
        clf_eval = ClassificatonModelEvaluation(model_all_preds)
        eval_table, conf_matrix_path = clf_eval.evaluate_binary_multiclass_classfication_models(
            save_figure_path=project_path+'baseline_rf_conf_matrix.jpg')
        
        f1 = float(eval_table[eval_table['class-label/metric'] == "weighted avg"]['f1-score'].values)
        precision = float(eval_table[eval_table['class-label/metric'] == "weighted avg"]['precision'].values)
        recall = float(eval_table[eval_table['class-label/metric'] == "weighted avg"]['recall'].values)
    
        return_json = {
                'Status':'Done', 
                'Evauation_Table': eval_table.to_json(), 
                'Confusion_Matrix_Path': conf_matrix_path,
                'F1': f1,
                'Precision': precision,
                'Recall': recall,
                'Model_Object_Path': project_path+'baseline_rf_model.pk',
                'Model_Parameters': str(models_list[0].get_params())
                }
    except Exception as e:
        return_json = {
                'Status':'Error : ' + str(e), 
            }
    
    return json.dumps(return_json)



@app.route('/tuned_model_train_rf/')
def tuned_model_train_rf():
    try:
        json_data = json.loads(unquote(request.args.get('json')))
    
        data_path = json_data['data_path']                 
        data_dtypes_path = json_data['data_dtypes_path']
        target_variable = json_data['target_variable']
        project_path = json_data['project_path']
    
        data = DetectColumnTypes().read_data(data_path)
        my_dtypes = pd.read_csv(data_dtypes_path)
    
        base_data_prep = BaselineDataPrep(data, my_dtypes)
        data = base_data_prep.get_baseline_prepared_data(one_hot_encode=True, target=target_variable)
    
        X_train = data[data.columns.difference([target_variable])]
        Y = data[target_variable]
    
    
        rf = RandomForestClassifier()
        search_spaces_rf = {
            'class_weight': ['balanced', None],
            'max_features': (2, 10),
            'max_depth': (3, 50),
            'n_estimators': (10, 20),
            'min_samples_leaf': (10, 100),
            'min_samples_split': (5, 30),
            'min_weight_fraction_leaf': (0.01, 0.5, 'uniform')
        }

        cval_model_builder = CrossValModelBuilder(model_object=rf, target_variable=Y)
        tuned_model, tuned_model_preds = cval_model_builder.fit_bayesian_tuning_model(X_train, Y, search_spaces_rf, 
                                    project_path+'tuned_model_train_rf_log.txt', project_path)
    
        joblib.dump(tuned_model, project_path+'tuned_rf_model.pk')
    
        clf_eval = ClassificatonModelEvaluation(tuned_model_preds)
        eval_table, conf_matrix_path = clf_eval.evaluate_binary_multiclass_classfication_models(
            save_figure_path=project_path+'tuned_rf_conf_matrix.jpg')
        
        f1 = float(eval_table[eval_table['class-label/metric'] == "weighted avg"]['f1-score'].values)
        precision = float(eval_table[eval_table['class-label/metric'] == "weighted avg"]['precision'].values)
        recall = float(eval_table[eval_table['class-label/metric'] == "weighted avg"]['recall'].values)
    
        return_json = {
                'Status':'Done', 
                'Evauation_Table': eval_table.to_json(), 
                'Confusion_Matrix_Path': conf_matrix_path,
                'F1': f1,
                'Precision': precision,
                'Recall': recall,
                'Model_Object_Path': project_path+'tuned_rf_model.pk', 
                'Model_Parameters': str(tuned_model.get_params())
                }
    except Exception as e:
        return_json = {
                'Status':'Error : ' + str(e), 
            }
    
    return json.dumps(return_json)



#####################################################################################

@app.route('/baseline_linear_regression/')
def baseline_linear_regression():
    try:
        json_data = json.loads(unquote(request.args.get('json')))
    
        data_path = json_data['data_path']                 
        data_dtypes_path = json_data['data_dtypes_path']
        target_variable = json_data['target_variable']
        project_path = json_data['project_path']
    
        data = DetectColumnTypes().read_data(data_path)
        my_dtypes = pd.read_csv(data_dtypes_path)
    
        base_data_prep = BaselineDataPrep(data, my_dtypes)
        RegData = base_data_prep.get_baseline_prepared_data(one_hot_encode=True, target=target_variable)
    
        X_train = data[data.columns.difference([target_variable])]
        Y = data[target_variable]
    
        ModelObj, R_Square, RMSE, MAE, predicted_vs_actuals, ModelCoefs = Linear_Model1(RegData, target_variable)
        
        joblib.dump(ModelObj, project_path+'baseline_linear_regression.pk')
    
        return_json = {
                'Status':'Done', 
                'R_Square': round(R_Square,2), 
                'RMSE': round(RMSE,2),
                'MAE': round(MAE,2),
                'Model_Object_Path': project_path+'baseline_linear_regression.pk',
                'Model_Parameters': str(ModelObj.get_params()),
                'Model_Coefficients': ModelCoefs.to_json()
            }
    except Exception as e:
        return_json = {
                'Status':'Error : ' + str(e), 
            }
    
    return json.dumps(return_json)


@app.route('/tuned_elastic_net_model/')
def tuned_elastic_net_model():
    try:
        json_data = json.loads(unquote(request.args.get('json')))
    
        data_path = json_data['data_path']                 
        data_dtypes_path = json_data['data_dtypes_path']
        target_variable = json_data['target_variable']
        project_path = json_data['project_path']
    
        data = DetectColumnTypes().read_data(data_path)
        my_dtypes = pd.read_csv(data_dtypes_path)
    
        base_data_prep = BaselineDataPrep(data, my_dtypes)
        RegData = base_data_prep.get_baseline_prepared_data(one_hot_encode=True, target=target_variable)
    
        X_train = data[data.columns.difference([target_variable])]
        Y = data[target_variable]
    
        ModelObj, R_Square, RMSE, MAE, predicted_vs_actuals, ModelCoefs = ElasticNet_Model1(RegData, target_variable)
        
        joblib.dump(ModelObj, project_path+'tuned_elastic_net_model.pk')
    
        return_json = {
                'Status':'Done', 
                'R_Square': round(R_Square,2), 
                'RMSE': round(RMSE,2),
                'MAE': round(MAE,2),
                'Model_Object_Path': project_path+'tuned_elastic_net_model.pk',
                'Model_Parameters': str(ModelObj.get_params()),
                'Model_Coefficients': ModelCoefs.to_json()
            }
    except Exception as e:
        return_json = {
                'Status':'Error : ' + str(e), 
            }
    
    return json.dumps(return_json)


@app.route('/baseline_model_train_rf_regression/')
def baseline_model_train_rf_regression():
    try:
        json_data = json.loads(unquote(request.args.get('json')))
    
        data_path = json_data['data_path']                 
        data_dtypes_path = json_data['data_dtypes_path']
        target_variable = json_data['target_variable']
        project_path = json_data['project_path']
    
        data = DetectColumnTypes().read_data(data_path)
        my_dtypes = pd.read_csv(data_dtypes_path)
    
        Base_line = BaselineDataPrep(data, my_dtypes)
        Base_line_data = Base_line.get_baseline_prepared_data(one_hot_encode=True)
        #Base_line_data = Base_line_data[data.columns.intersection(data_type[data_type.Type.isin(['Numeric'])].ColumName)]
        x = Base_line_data.drop(target_variable, axis=1)
        y = Base_line_data[target_variable]
    
        params = {
            'verbose':50,
            'bootstrap': True,
            'n_estimators': 30, 
            'random_state':42
        }
        rf = RandomForestRegressor(**params)
        cval_model_builder_regression = CrossValModelBuilderRegression(model_object=rf, target_variable=target_variable)
        models_list, predicted_vs_actuals = cval_model_builder_regression.fit_base_cv_model(x, y,
                                    project_path+'baseline_model_train_rf_regression_log.txt')
    
    
        ModelObj = models_list[0]
        
        joblib.dump(ModelObj, project_path+'baseline_rf_regression.pk')
    
        R_Square, RMSE, MAE = score_regression_models(predicted_vs_actuals)
        
        return_json = {
                'Status':'Done', 
                'R_Square': round(R_Square,2), 
                'RMSE': round(RMSE,2),
                'MAE': round(MAE,2),
                'Model_Object_Path': project_path+'baseline_rf_regression.pk',
                'Model_Parameters': str(ModelObj.get_params()),
            }
    except Exception as e:
        return_json = {
                'Status':'Error : ' + str(e), 
            }
    
    return json.dumps(return_json)


@app.route('/tuned_model_train_rf_regression/')
def tuned_model_train_rf_regression():
    try:
        json_data = json.loads(unquote(request.args.get('json')))
    
        data_path = json_data['data_path']                 
        data_dtypes_path = json_data['data_dtypes_path']
        target_variable = json_data['target_variable']
        project_path = json_data['project_path']
    
        data = DetectColumnTypes().read_data(data_path)
        my_dtypes = pd.read_csv(data_dtypes_path)
    
        Base_line = BaselineDataPrep(data, my_dtypes)
        Base_line_data = Base_line.get_baseline_prepared_data(one_hot_encode=True)
        #Base_line_data = Base_line_data[data.columns.intersection(data_type[data_type.Type.isin(['Numeric'])].ColumName)]
        x = Base_line_data.drop(target_variable, axis=1)
        y = Base_line_data[target_variable]
    
        rf = RandomForestRegressor()
        search_spaces_rf = {
            'max_features': (0, 3),
            'max_depth': (3, 50),
            'n_estimators': (10, 200),
            'min_samples_leaf': (10, 100),
            'min_samples_split': (5, 30),
            'min_weight_fraction_leaf': (0.01, 0.5, 'uniform')
        }
        cval_model_builder_regression = CrossValModelBuilderRegression(model_object=rf, target_variable=target_variable)
        ModelObj, predicted_vs_actuals = cval_model_builder_regression.fit_bayesian_tuning_model(x, y, search_spaces_rf, 
                                    project_path+'tuned_model_train_rf_regression_log.txt', 
                                    project_path)
        
        joblib.dump(ModelObj, project_path+'tuned_model_train_rf_regression.pk')
    
        R_Square, RMSE, MAE = score_regression_models(predicted_vs_actuals)
        
        return_json = {
                'Status':'Done', 
                'R_Square': round(R_Square,2), 
                'RMSE': round(RMSE,2),
                'MAE': round(MAE,2),
                'Model_Object_Path': project_path+'tuned_model_train_rf_regression.pk',
                'Model_Parameters': str(ModelObj.get_params()),
            }
    except Exception as e:
        return_json = {
                'Status':'Error : ' + str(e), 
            }
    
    return json.dumps(return_json)

#####################################################################################



if __name__ == '__main__':
    app.run(debug=True)