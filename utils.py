

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
