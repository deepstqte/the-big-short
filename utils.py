from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import re

def na_catfiller(df):
    """
    This function identifies categorical and Boolean features,
    then replaces missing values with "missing_data" value.
    :param df:
    """
    objectCols = []
    for i, j in zip(df.dtypes.index, df.dtypes.values):
        if j == "object":
            objectCols.append(i)
    booleanCols = []
    for i, j in zip(df.dtypes.index, df.nunique()):
        if j == 2:
            booleanCols.append(i)

    df[objectCols] = df[objectCols].fillna("missing_data")
    df[booleanCols] = df[booleanCols].fillna("missing_data")
    return df

def na_numfiller(df, aggregation_func="mean"):
    """
    This function identifies non-categorical features, then identifies
     non-Boolean features within the resulting list of features, and
     then identifies features with missing values within the
     resulting list of features.
     It then returns the list of features identified. Lastly,
     it fills missing values identified within those features with
     the method selected by the user.
    :param df:
    :param aggregation_func:
    """
    non_objectCols = []
    for i, j in zip(df.dtypes.index, df.dtypes.values):
        if j != "object":
            non_objectCols.append(i)

    non_BooleanCols = []
    for i, j in zip(df[non_objectCols].nunique().index, df[non_objectCols].nunique().values):
        if j > 2:
            non_BooleanCols.append(i)

    null_cols = []
    for i, j in zip(df[non_BooleanCols].isnull().sum().index, df[non_BooleanCols].isnull().sum().values):
        if j > 0:
            null_cols.append(i)
    if aggregation_func == "mean":
        df.fillna(df[null_cols].mean(), inplace=True)
    elif aggregation_func == "median":
        df.fillna(df[null_cols].median(), inplace=True)
    else:
        return False
    return df

def str_one_hot_encoder(df, unique_threshold=10):
    str_columns = df.select_dtypes(include=['object']).columns
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace(',', '_')
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    return_df = df
    for column in str_columns:
        if df[column].nunique() <= unique_threshold:
            df_dummies = pd.get_dummies(df[column],prefix=column)
            return_df = pd.concat([return_df, df_dummies], axis=1)
        return_df.drop(column, axis=1, inplace=True)
    return_df = return_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    return return_df

def aggr_dicts_json_to_dict(data_from_json):
    custom_merge_aggr_funcs = {
        "max_min_diff": lambda x: (x.max()-x.min(), "max_min_diff")
    }
    aggr_dicts = {}
    for df, df_features in data_from_json.items():
        aggr_dicts[df] = {}
        for feature, aggr_funcs in df_features.items():
            aggr_dicts[df][feature] = []
            for func in aggr_funcs:
                if func in custom_merge_aggr_funcs:
                    aggr_dicts[df][feature].append(custom_merge_aggr_funcs[func])
                else:
                    aggr_dicts[df][feature].append(func)
    return aggr_dicts

def str_catencoder(df, method_switch=10):
    """
    This function applies one-hot encoding or label encoding on
    categorical string colmns based on the number of unique values
    in the given dataframe.
    One-hot encoding is used if thenumber of unique values is smaller
    or equal to the "method_switch" threshold, and label encoding is
    used if bigger.
    :param df:
    :param method_switch:
    """
    le = LabelEncoder()
    str_columns = df.select_dtypes(include=['object']).columns
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace(',', '_')
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    return_df = df
    for column in str_columns:
        if df[column].nunique() <= method_switch:
            df_dummies = pd.get_dummies(df[column],prefix=column)
            return_df = pd.concat([return_df, df_dummies], axis=1)
            return_df.drop(column, axis=1, inplace=True)
        else:
            return_df[column] = return_df[column].map(str)
            return_df[column] = le.fit_transform(return_df[column])
    return_df = return_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    return return_df

def merge_with_aggr(main_df, secondary_df, fk_column, aggr_dict, column_prefix):
    if not aggr_dict:
        return main_df
    secondary_df = secondary_df.groupby(fk_column).agg(aggr_dict)
    columns_list = []
    aggr_df = pd.DataFrame()
    for x in secondary_df.columns.tolist():
        if x[1].startswith("<lambda") and secondary_df[x[0]][x[1]].dtypes == "object" and len(secondary_df[x[0]][x[1]].iloc[0]) > 1:
            column_name = '{}_{}_{}'.format(column_prefix,x[0],secondary_df[x[0]][x[1]].iloc[0][1])
            aggr_df[column_name] = secondary_df[x[0]][x[1]].apply(lambda x: x[0])
        else:
            column_name = '{}_{}_{}'.format(column_prefix,x[0],x[1])
            aggr_df[column_name] = secondary_df[x[0]][x[1]]
    aggr_df = aggr_df.rename_axis(secondary_df.index.name)
    return main_df.merge(aggr_df,on=fk_column,how='left')

### OPTIMIZING DATA TYPES ###

# Create function to optimize integer data types
def optimize_inttypes(dataframe, specify="auto"):
    # Construct dataframe for reference used below in the optimize_inttypes function.
    np_types = [np.int8 ,np.int16 ,np.int32, np.int64,
               np.uint8 ,np.uint16, np.uint32, np.uint64]
    np_types = [np_type.__name__ for np_type in np_types]
    type_df = pd.DataFrame(data=np_types, columns=['class_type'])
    type_df['min_value'] = type_df['class_type'].apply(lambda row: np.iinfo(row).min)
    type_df['max_value'] = type_df['class_type'].apply(lambda row: np.iinfo(row).max)
    type_df['range'] = type_df['max_value'] - type_df['min_value']
    type_df.sort_values(by='range', inplace=True)
    # Print initial memory usage details
    mem_sum = 0
    print('Memory usage of dataframe is {:.6f} GB'.format(dataframe.memory_usage().sum()/1000000000))
    mem_sum = mem_sum+dataframe.memory_usage().sum()/1000000000
    print(f"Total initial memory used for selected dataframe is: {mem_sum:.2f}GB")
    if specify=="auto":
        for col in dataframe.loc[:, dataframe.dtypes <= np.int64]:
            col_min = dataframe[col].min()
            col_max = dataframe[col].max()
            temp = type_df[(type_df['min_value'] <= col_min) & (type_df['max_value'] >= col_max)]
            optimized_class = temp.loc[temp['range'].idxmin(), 'class_type']
            print("Col name : {} Col min_value : {} Col max_value : {} Optimized Class : {}".format(col, col_min, col_max, optimized_class))
            dataframe[col] = dataframe[col].astype(optimized_class)
    elif specify=="int32":
        for col in dataframe.loc[:, dataframe.dtypes == np.int64]:
            dataframe[col] = dataframe[col].astype(np.int32)
        # Print updated memory usage details
        mem_sum = 0
        print('Memory usage of dataframe is {:.6f} GB'.format(dataframe.memory_usage().sum()/1000000000))
        mem_sum = mem_sum+dataframe.memory_usage().sum()/1000000000
        print(f"Total memory currently used for selected dataframe is: {mem_sum:.2f}GB")
    else:
        return False
    return

####################################################################################################
####################################################################################################



# Create float optimization function.
def optimize_floattypes(dataframe, specify="auto"):
    # Create dataframe for reference, used in the float optimization function further below.
    np_types = [np.float16 ,np.float32, np.float64]
    np_types = [np_type.__name__ for np_type in np_types]
    floattype_df = pd.DataFrame(data=np_types, columns=['class_type'])
    floattype_df['min_value'] = floattype_df['class_type'].apply(lambda row: np.finfo(row).min)
    floattype_df['max_value'] = floattype_df['class_type'].apply(lambda row: np.finfo(row).max)
    floattype_df['range'] = floattype_df['max_value'] - floattype_df['min_value']
    floattype_df.sort_values(by='range', inplace=True)# Print initial memory usage details
    mem_sum = 0
    print('Memory usage of dataframe is {:.6f} GB'.format(dataframe.memory_usage().sum()/1000000000))
    mem_sum = mem_sum+dataframe.memory_usage().sum()/1000000000
    print(f"Total initial memory used for selected dataframe is: {mem_sum:.2f}GB")
    if specify == "auto":
        for col in dataframe.loc[:, dataframe.dtypes == np.float64]:
            col_min = dataframe[col].min()
            col_max = dataframe[col].max()
            temp = floattype_df[(floattype_df['min_value'] <= col_min) & (floattype_df['max_value'] >= col_max)]
            optimized_class = temp.loc[temp['range'].idxmin(), 'class_type']
            print("Col name : {} Col min_value : {} Col max_value : {} Optimized Class : {}".format(col, col_min, col_max, optimized_class))
            dataframe[col] = dataframe[col].astype(optimized_class)
        # Print updated memory usage details
        mem_sum = 0
        print('Memory usage of dataframe is {:.6f} GB'.format(dataframe.memory_usage().sum()/1000000000))
        mem_sum = mem_sum+dataframe.memory_usage().sum()/1000000000
        print(f"Total memory currently used for selected dataframe is: {mem_sum:.2f}GB")
    elif specify == "float32":
        for col in dataframe.loc[:, dataframe.dtypes == np.float64]:
            dataframe[col] = dataframe[col].astype(np.float32)
        # Print updated memory usage details
        mem_sum = 0
        print('Memory usage of dataframe is {:.6f} GB'.format(dataframe.memory_usage().sum()/1000000000))
        mem_sum = mem_sum+dataframe.memory_usage().sum()/1000000000
        print(f"Total memory currently used for selected dataframe is: {mem_sum:.2f}GB")
    else:
        return False
    return
