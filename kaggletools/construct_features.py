import pandas as pd

def get_most_correlated_column(data, target, columns=None):
    """
    Builds a linear combination of columns of the source data frame, best correlated with
    the target column.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The Pandas data frame containing source features.
   
    target : pandas.Series
        The target series.
    
    columns : list, optional
        If given, only use features listed here
    
    Returns
    -------
    pandas.Series
        The linear combination of `data` columns, most correlated with the
        target.
    """
    if columns:
        active_cols = set(data.columns) & set(columns)
    else:
        active_cols = data.columns
    approximator = pd.Series(0, index=target.index)
    for x in active_cols:
        approximator += data[x] * target.corr(data[x])
    return approximator
