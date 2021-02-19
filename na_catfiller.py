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