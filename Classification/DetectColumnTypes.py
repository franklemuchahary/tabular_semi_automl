import pandas as pd
import numpy as np
from datetime import datetime
import dateutil.parser as parse_date
from random import randint
import csv

class DetectColumnTypes():
    def __init__(self):
        self.data = ''
        self.column_data_types = {}
        self.secondary_column_tags = {}
        
    def read_data(self, load_file_path=''):
        try:
            single_line_file = []
            f = open(load_file_path,"r")
            for line in f.readlines():
                single_line_file.append(line)   
                break
            f.close()

            csv_sniffer = csv.Sniffer()
            delim = csv_sniffer.sniff(single_line_file[0]).delimiter
            
            self.data = pd.read_csv(load_file_path, sep=",")
            return self.data
        except Exception as e:
            pass
            #er.log_error(e, 'error_logs/read_file_errors.txt', write_mode='w')
    
    def _detect_date_time_type_columns(self):
        for col in self.data.columns.values:
            try:
                for _ in range(10):
                    data_to_test = self.data[col][randint(1, len(self.data)-1)]
                    if data_to_test!='' or data_to_test!=None: 
                        parse_date.parse(data_to_test)
                self.column_data_types[col] = ["DateTime"]
            except Exception as e:
                pass
            
    def _detect_categorical_columns(self):
        '''
        _detect_date_time_type_columns method needs to be called before this to get datetime correctly
        '''
        for col in self.data.columns.values:
            date_col_check = self.column_data_types.get(col)
            categorical_check_condition = (self.data[col].nunique()/self.data[col].count()) < 0.05
            categorical_check_condition2 = self.data[col].dtype == object
            categorical_check_condition3 = self.data[col].dtype == float
            if (categorical_check_condition | categorical_check_condition2) and (date_col_check!=["DateTime"]) and (categorical_check_condition3 != True):
                self.column_data_types[col] = ["Categorical"]
            else:
                if (date_col_check!=["DateTime"]):
                    self.column_data_types[col] = ["Numeric"]
                    
    def _detect_id_columns(self):
        data = self.data
        for col in data.columns.values:
            unique_of_values = data[col].nunique()/len(data[col])
            is_not_float = data[col].dtype != float
            if ('id' in col.lower() and is_not_float==True) and (unique_of_values > 0.95 and  is_not_float==True):
                self.secondary_column_tags[col] = ["Potential ID Column"]
                
                
    def get_column_types(self, load_file_path='', write_file_path=''):
        _ = self.read_data(load_file_path)
        self._detect_date_time_type_columns()
        self._detect_categorical_columns()
        self._detect_id_columns()
        types = self.column_data_types
        secondary_types = self.secondary_column_tags
    
        types = pd.DataFrame(types)
        secondary_types = pd.DataFrame(secondary_types)
        dtypes_df = pd.concat([types,secondary_types]).T.reset_index()
        
        if len(dtypes_df.columns)==3:
            dtypes_df.columns = ['ColumName', 'Type', 'SecondaryType']
        else:
            dtypes_df.columns = ['ColumName', 'Type']
        
        dtypes_df.to_csv(write_file_path, index=False)
        
        return dtypes_df