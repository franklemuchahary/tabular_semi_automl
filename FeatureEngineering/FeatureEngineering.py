import pandas as pd
import numpy as np

def get_count_and_mode_cols(data, my_dtypes, target_variable):
    unique_count_cols =[]

    for col in data.columns.values:
        dtype = my_dtypes[my_dtypes['ColumName'] == col]['Type'].values[0]
        if data[col].nunique() > 5 and data[col].nunique() < 50 and col!=target_variable and dtype=="Categorical":
            unique_count_cols.append(col)
            
    mode_cols = unique_count_cols
    
    return (unique_count_cols, mode_cols)

def get_mean_cols(data, my_dtypes, target_variable):
    mean_cols =[]

    for col in data.columns.values:
        dtype = my_dtypes[my_dtypes['ColumName'] == col]['Type'].values[0]
        if col!=target_variable and dtype=="Numeric" and ('id' not in col.lower()) :
            mean_cols.append(col)
            
    return mean_cols


def generate_agg_features(concat_df, data_dtypes, group_by_cols, target_variable):
    mean_cols = get_mean_cols(concat_df, data_dtypes, target_variable)
    unique_count_cols, mode_cols = get_count_and_mode_cols(concat_df, data_dtypes, target_variable)
    
    for gp_by_col in group_by_cols:
        unique_count_cols = [i for i in unique_count_cols if i!=gp_by_col]
        mode_cols = [i for i in mode_cols if i!=gp_by_col]
        
        if len(mean_cols)>0:
            gp_values_mean = concat_df.groupby(gp_by_col)[mean_cols].mean()
            gp_values_mean.columns = ['mean_feat_by_'+gp_by_col + '_' + i for i in gp_values_mean.columns.values]
            gp_values_mean.reset_index(inplace=True)
            concat_df = pd.merge(concat_df, gp_values_mean, how='left', on=gp_by_col)
        
        if len(unique_count_cols):
            gp_values_count = concat_df.groupby(gp_by_col)[unique_count_cols].nunique()
            gp_values_count.columns = ['distinct_count_feat_by_'+gp_by_col + '_' + i 
                                       for i in gp_values_count.columns.values]
            gp_values_count.reset_index(inplace=True)
            concat_df = pd.merge(concat_df, gp_values_count, how='left', on=gp_by_col)
        
            gp_values_mode = concat_df.groupby(gp_by_col)[mode_cols].agg(lambda x:x.value_counts().index[0])
            gp_values_mode.columns = ['mode_feat_by_'+gp_by_col + '_' + i for i in gp_values_mode.columns.values]
            gp_values_mode.reset_index(inplace=True)
            concat_df = pd.merge(concat_df, gp_values_mode, how='left', on=gp_by_col)        
        
    return concat_df


def get_feature_engieering_variables(data, data_dtypes, target_variable):
    cols_to_group_on = []

    for col in data.columns.values:
        dtype = data_dtypes[data_dtypes['ColumName'] == col]['Type'].values[0]
        if data[col].nunique() > 3 and data[col].nunique() < 30  and col!=target_variable and dtype=="Categorical":
            cols_to_group_on.append(col)
            
    if len(cols_to_group_on) > 0:
        feature_engineered_df = generate_agg_features(data, data_dtypes, cols_to_group_on, target_variable)
    else:
        feature_engineered_df = data
        
        
    new_cols = feature_engineered_df.columns.difference(data.columns.values).values 
    
    if len(new_cols)>0:
        final_cols = []
        for i in new_cols:
            if np.std(feature_engineered_df[i]) > 0.1:
                final_cols.append(i)
        
        
        all_columns = list(data.columns.values)+final_cols
        number_of_features_engineered = len(final_cols)
    else:
        all_columns = data.columns.values
    
    final_feature_engineered_df = feature_engineered_df[all_columns]  
        
    return (final_feature_engineered_df, all_columns, number_of_features_engineered)