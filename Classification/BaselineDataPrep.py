from sklearn import preprocessing
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class BaselineDataPrep():
    def __init__(self, my_data, data_detected_dtypes):
        self.data = my_data
        self.detected_dtypes = data_detected_dtypes
    
    def _label_encoding_func(self, df_name, df_col_name):
        '''
        label encoder method
        '''
        le = preprocessing.LabelEncoder()
        le.fit(df_name[df_col_name])
        return le.transform(df_name[df_col_name])

    def _do_one_hot_encoding(self, df_name, df_column_name, prefix=''):
        '''
        one hot encoder method
        '''
        x = pd.get_dummies(df_name[df_column_name], prefix=prefix)
        df_name = df_name.join(x)
        df_name = df_name.drop(df_column_name, axis=1) 
        return df_name
    
    def _variable_encoder(self, data, data_dtypes, one_hot_encode=False, one_hot_threshold=10, target=''):
        '''
        automatic variable encoder -> calls the above two methods automatically
        '''
        data_types = data.dtypes
    
        for col in data_types.index.values:
            col_type = data_dtypes[data_dtypes['ColumName'] == col]['Type'].values[0]
            if col_type == "Categorical":
                if one_hot_encode == True and data[col].nunique() < one_hot_threshold and col!=target:
                    data = self._do_one_hot_encoding(data, col, col)
                else:
                    if data_types[col] == object:
                        data[col] = self._label_encoding_func(data, col)
            
            if col_type == "DateTime" and data_types[col] == object:
                data[col] = pd.to_datetime(data[col])
                data[col] = self._label_encoding_func(data, col)
    
        return data
    
    def _fill_nulls(self, data, data_dtypes):
        '''
        automatic null values filler
        '''
        null_cols = data.isnull().sum()
        null_cols = null_cols[null_cols>0]
    
        for col in null_cols.index.values:
            col_type = data_dtypes[data_dtypes['ColumName'] == col]['Type'].values[0]
            if col_type == "Categorical":
                if data[col].dtype == object:
                    data[col].fillna("UNKNOWN", inplace=True)
                else:
                    data[col].fillna(-100, inplace=True)
    
            if col_type == "Numeric":
                data[col].fillna(data[col].mean(), inplace=True)
        
            if col_type == "DateTime":
                data[col].fillna(data[col].value_counts().index.values[0], inplace=True)
            
        return data
    
    def get_baseline_prepared_data(self, one_hot_encode=False, one_hot_threshold=10, target=''):
        self.data = self._fill_nulls(self.data, self.detected_dtypes)
        self.data = self._variable_encoder(self.data, self.detected_dtypes, 
                                              one_hot_encode, one_hot_threshold, target=target)
        
        return self.data