import numpy as np
from sklearn.metrics import log_loss, accuracy_score

from evaluation.purged_cv import PurgedKFold


class Metrics:
    NEG_LOG_LOSS = 'neg_log_loss'
    ACCURACY = 'accuracy'


def timeseries_cv_score(
        clf, X, y, sample_weight, scoring=Metrics.NEG_LOG_LOSS,
        event_times=None, cv=None, cv_splitter=None, embargo_pct=None
):
    """
    Perform cross validation scoring on time series data. See also: sklearn cross_val_score

    :param clf: the classifier model
    :param X: input features DataFrame
    :param y: label Series
    :param sample_weight: array of sample weights
    :param scoring: scoring function name
    :param event_times: used for PurgedKFold. Can be None if cv_splitter is provided
    :param cv: number of cv folds
    :param cv_splitter: default to using PurgedKFold if None
    :param embargo_pct: embargo parameter for PurgedKFold
    :return: an array of score for each CV split. Can be None if cv_splitter is provided
    """

    if scoring not in ['neg_log_loss', 'accuracy']:
        raise NotImplementedError(f'scoring: {scoring}')
    if cv_splitter is None:
        cv_splitter = PurgedKFold(n_splits=cv, event_times=event_times, embargo_pct=embargo_pct)

    scores = []
    for train_indices, test_indices in cv_splitter.split(X):
        X_train = X.iloc[train_indices, :]
        y_train = y.iloc[train_indices]
        train_sample_weight = sample_weight.iloc[train_indices].values
        X_test = X.iloc[test_indices, :]
        y_test = y.iloc[test_indices]
        test_sample_weight = sample_weight.iloc[test_indices].values

        fitted = clf.fit(X=X_train, y=y_train, sample_weight=train_sample_weight)

        if scoring == Metrics.NEG_LOG_LOSS:
            prob = fitted.predict_proba(X_test)
            split_score = -log_loss(y_test, prob, sample_weight=test_sample_weight, labels=clf.classes_)
        else:
            pred = fitted.predict(X_test)
            split_score = accuracy_score(y_test, pred, sample_weight=test_sample_weight)

        scores.append(split_score)

    return np.array(scores)


def sharpe_ratio(returns_series: np.ndarray, axis=None):
    """
    :param returns_series: array of returns or other performance metrics e.g. PnL
    :param axis: calculate sharpe ration along this axis
    :return: Sharpe ratio
    """
    return np.mean(returns_series, axis=axis)/np.std(returns_series, axis=axis)
