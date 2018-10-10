"""
Functions anlyzing "Titanic" data set, the training set for beginners
in machine learning. See `https://www.kaggle.com/c/titanic` for more
info.
"""

def extract_title(data, titles=None):
    """
    Create a title column meaning whether the `data.Name` column
    contains Mr, Mrs, Miss, etc.title. Returns a Series of
    integers indicating the titles. Default mapping is the following.
    
    *   0 means Mr
    *   1 means Mrs
    *   2 means Miss
    *   3 means Master
    *   4 means rare titles like Dr, Sir, etc.
    
    This mapping may be customized using the `titles` argument
    (see below).
    
    Parameters
    ----------
    data : pandas.DataFrame
        The data frame that must contain the 'Name' column containing
        the 'Mr', 'Mrs', etc. title in addition to the name itself.
    
    titles : list, default ['Mr', 'Mrs', 'Miss', 'Master', 'Rare']
        The list of interesting titles. It must contain the Mr, Mrs,
        Miss, Master and Rare strings and may contain Dr, Military
        and Royal. For every passenger, the returned column will
        contain the integer meaning index of this passenger's title
        in the list.

    Returns
    -------
    pandas.Series
        The list of integers of range 0-4 (or `0` to `len (titles) - 1` if
        `titles` argument given) indicating the titles taken from the name
        column. The series index will be same with that of data argument.
    
    Issues
    ------
    *   If `Rare` title not given as interesting, it may be a good idea
        to use 'Mr', 'Mrs', 'Miss', or 'Master' titles depending on
        gender and age.
    """
    if titles is None:
        titles = ['Mr', 'Mrs', 'Miss', 'Master', 'Rare']
    def _mapper(x):
        if x in ["Mr"]:
            return titles.index("Mr")
        elif x in ["Mrs", "Mme"]:
            return titles.index("Mrs")
        elif x in ["Miss", "Ms", "Mlle"]:
            return titles.index("Miss")
        elif x in ["Master"]:
            return titles.index("Master")
        elif x in ["Dr"] and "Dr" in titles:
            return titles.index("Dr")
        elif x in ["Capt", "Major", "Col"] and "Military" in titles:
            return titles.index("Military")
        elif x in ["Sir", "Count", "Countess"] and "Royal" in titles:
            return titles.index("Royal")
        else:
            return titles.index("Rare")
    return data.Name.str.extract("([A-Za-z]+)\\.", expand=False).map(_mapper)
