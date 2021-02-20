from sklearn.preprocessing import LabelEncoder

def na_catfiller(df):
    """
    This function identifies categorical features, then replaces
    missing values with "unknown" value.
    :param df:
    """
    objectCols = []
    for i, j in zip(df.dtypes.index, df.dtypes.values):
        if j == "object":
            objectCols.append(i)
    df[objectCols] = df[objectCols].fillna("unknown")
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
    return 

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
            df_dummies = pd.get_dummies(dfs["application_train"][column],prefix=column)
            return_df = pd.concat([return_df, df_dummies], axis=1)
            return_df.drop(column, axis=1, inplace=True)
        else:
            return_df[column] = return_df[column].map(str)
            return_df[column] = le.fit_transform(return_df[column])
    return return_df
