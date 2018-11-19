"""
Select best subset of features to fit the model on. So far for regression
models only.
"""

from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit

# Needed to conserve memory
import gc

def select_features_ascending(data, y, model, verbose=False):
    """
    Selects the best features for model fitting. First, selects the first
    feature providing the best-fitted model on this feature only. Then,
    selects the second feature to provide the best-fitted model on
    the two features and so on. Stops if no more features improve the
    model.

    Parameters
    ----------
    data : pandas.DataFrame
        The source feature matrix to fit the model on. It must be a data
        frame, not a Numpy array, because every feature should have a name.

    y : pandas.Series or numpy.ndarray
        The source target array

    model : sklearn.Estimator or compatible (e.g. XGBRegressor)
        The model to be fitted. Currently must be a sklearn-compatible
        regressor.

    verbose : bool, optional
        Display progress information

    Returns
    -------
    list :
        The list of column names to fit the model on.

    Known bugs
    ----------
    It would be good to have a `verbose` option. By default, much debugging
    output is written to stdout and this is not always good.
    """
    all_features = list(data.columns)
    selected_features = []
    best_score = None
    best_score_str = None
    for n in range(0, len(all_features)):
        best_features = None
        for f in set(all_features) - set(selected_features):
            try_features = selected_features + [f]
            cv = cross_validate(estimator=model,
                                X=data[try_features].values,
                                y=y,
                                scoring='r2',
                                return_train_score=True,
                                cv=ShuffleSplit(n_splits=10,
                                                random_state=0,
                                                train_size=0.75,
                                                test_size=0.25),
                                n_jobs=-1)
            score = cv['test_score'].mean()
            score_str = "%lf +/- %lf" % (cv['test_score'].mean(), 3 * cv['test_score'].std())
            if verbose:
                print("Features: %s" % str(try_features))
                print("  Testing score: %s" % score_str)
            if best_score is None or score > best_score:
                best_features = try_features
                best_score = score
                best_score_str = score_str
        if best_features:
            selected_features = best_features
            if verbose:
                print("Currently selected features: %s" % str(best_features))
                print("Best score: %s" % best_score_str)
        else:
            if verbose:
                print("Best features: %s" % str(selected_features))
                print("Best score: %s" % best_score_str)
            break
        gc.collect()
    return selected_features

def select_features_descending(data, y, model, verbose=False):
    """
    Selects the best features for model fitting. First, fits the model on all
    features, then tries to remove every single feature while that can improve
    the model. That seems to work good, but much slower than
    `sklearn.feature_selection.RFECV`.

    Parameters
    ----------
    data : pandas.DataFrame
        The source feature matrix to fit the model on. It must be a data
        frame, not a Numpy array, because every feature should have a name.

    y : pandas.Series or numpy.ndarray
        The source target array

    model : sklearn.Estimator or compatible (e.g. XGBRegressor)
        The model to be fitted. Currently must be a sklearn-compatible
        regressor.

    verbose : bool, optional
        Display progress information

    Returns
    -------
    list :
        The list of column names to fit the model on.

    Known bugs
    ----------
    It would be good to have a `verbose` option. By default, much debugging
    output is written to stdout and this is not always good.
    """
    all_features = list(data.columns)
    selected_features = all_features
    cv = cross_validate(estimator=model,
                        X=data.values,
                        y=y,
                        scoring='r2',
                        return_train_score=True,
                        cv=ShuffleSplit(n_splits=10,
                                        random_state=0,
                                        train_size=0.75,
                                        test_size=0.25),
                        n_jobs=-1)
    best_score = cv['test_score'].mean()
    best_score_str = "%lf +/- %lf" % (cv['test_score'].mean(), 3 * cv['test_score'].std())
    for n in range(0, len(all_features)):
        best_features = None
        for f in set(selected_features):
            try_features = list(set(selected_features) - set([f]))
            cv = cross_validate(estimator=model,
                                X=data[try_features].values,
                                y=y,
                                scoring='r2',
                                return_train_score=True,
                                cv=ShuffleSplit(n_splits=10,
                                                random_state=0,
                                                train_size=0.75,
                                                test_size=0.25),
                                n_jobs=-1)
            score = cv['test_score'].mean()
            score_str = "%lf +/- %lf" % (cv['test_score'].mean(), 3 * cv['test_score'].std())
            if verbose:
                print("Features: %s" % str(try_features))
                print("  Testing score: %s" % score_str)
            if best_score is None or score > best_score:
                best_features = try_features
                best_score = score
                best_score_str = score_str
        if best_features:
            selected_features = best_features
            if verbose:
                print("Currently selected features: %s" % str(best_features))
                print("Best score: %s" % best_score_str)
        else:
            if verbose:
                print("Best features: %s" % str(selected_features))
                print("Best score: %s" % best_score_str)
            break
        gc.collect()
    return selected_features

def squash_rare(data, colname, threshold=150, rare_val='Rare'):
    """
    Squash a number of rare categories into a single category.

    Parameters
    ----------
    data : pandas.DataFrame
        The source feature matrix. It will be modified after this call, so
        be careful.

    colname : str
        The name of the categorical features

    threshold : int, default=150
        Treat all category values encountered less than this number
        of times as rare and squash them into a single category.
        In future versions, float values will be acceptable, meaning
        percentage of total number of rows of source data matrix.

    rare_val : str or int, default 'Rare'
        The name of the new category. The default value for it is 'Rare',
        that is good if category names are all strings. If they are integers,
        an integer value must be given, best not encountered among existing
        category values.
    """
    rares = data[colname].value_counts() < threshold
    data[colname] = data[colname].apply(lambda x: rare_val if rares[x] else x)
