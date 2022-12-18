import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

from financial_ml.evaluation.metrics import Metrics
from financial_ml.evaluation.cross_validation import timeseries_cv_score, PurgedKFold


def get_mean_decrease_impurity(model, feature_names: list[str]):
    """
    Get the in-sample impurity decrease contributed by each feature in tree-based models.
    This measure is subject to substitution effects where a feature's importance is diluted by other
    interchangeable features.

    This modifies sklearn RF's feature_importances_ which considers a feature not chosen in a tree
    as having 0 importance to set them as np.nan instead.

    :param model: an sklearn-like model
    :param feature_names: list of feature names
    :return: a DataFrame of MDI feature importances
    """
    tree_importances = {i: tree.feature_importances_ for i, tree in enumerate(model.estimators_)}
    tree_importances = pd.DataFrame.from_dict(tree_importances, orient='index')
    tree_importances.columns = feature_names
    tree_importances = tree_importances.replace(0, np.nan)
    importances = pd.concat({
        'mean': tree_importances.mean(),
        'std': tree_importances.std()*tree_importances.shape[0]**-0.5
    }, axis=1)
    importances /= importances['mean'].sum()
    return importances


def get_mean_decrease_accuracy(
        clf, X: pd.DataFrame, y: pd.Series, cv: int, sample_weight: pd.Series,
        event_times: pd.Series, embargo_pct: float, scoring=Metrics.NEG_LOG_LOSS
):
    """
    Get the out-of-sample (OOS) decrease of selected evaluation score contributed by each feature using purged kfold
    cross validation. This method is model agnostic, but still subject to substitution effects.

    :param clf: a classifier model with sklearn's Estimator interface
    :param X: input features DataFrame
    :param y: labels
    :param cv: number of cv splits
    :param sample_weight:
    :param event_times: Series of event times
    :param embargo_pct: embargo fraction
    :param scoring: evaluation scores. Only neg_log_loss and accuracy are supported
    :return: MDA feature importances DataFrame, scores for each CV fold
    """
    if scoring not in [Metrics.NEG_LOG_LOSS, Metrics.ACCURACY]:
        raise NotImplementedError(scoring)

    cv_splitter = PurgedKFold(n_splits=cv, embargo_pct=embargo_pct)
    scores = pd.Series(dtype=float)
    perm_scores = pd.DataFrame(columns=X.columns)

    for fold, (train_indices, test_indices) in enumerate(cv_splitter.split(event_times)):
        X_train = X.iloc[train_indices, :]
        y_train = y.iloc[train_indices]
        w_train = sample_weight.iloc[train_indices]
        X_test = X.iloc[test_indices, :]
        y_test = y.iloc[test_indices]
        w_test = sample_weight.iloc[test_indices]

        clf = clf.fit(X_train, y_train, sample_weight=w_train.values)

        if scoring == Metrics.NEG_LOG_LOSS:
            prob = clf.predict_proba(X_test)
            scores.loc[fold] = -log_loss(y_test, prob, sample_weight=w_test.values, labels=clf.classes_)
        else:
            pred = clf.predict(X_test)
            scores.loc[fold] = accuracy_score(y_test, pred, sample_weight=w_test.values)

        for j in X.columns:
            X_test_perm = X_test.copy(deep=True)
            np.random.shuffle(X_test_perm[j].values)
            if scoring == Metrics.NEG_LOG_LOSS:
                prob = clf.predict_proba(X_test_perm)
                perm_scores.loc[fold, j] = -log_loss(y_test, prob, sample_weight=w_test.values, labels=clf.classes_)
            else:
                pred = clf.predict(X_test_perm)
                perm_scores.loc[fold, j] = accuracy_score(y_test, pred, sample_weight=w_test.values)

    importances = (-perm_scores).add(scores, axis=0)
    if scoring == Metrics.NEG_LOG_LOSS:
        importances /= -perm_scores  # Improvement relative to 0 log loss: (score - perm_score)/(0 - perm_score)
    else:
        importances /= (1. - perm_scores)  # Improvement relative to 1. accuracy: (score - perm_score)/(1 - perm_score)
    importances = pd.concat({
        'mean': importances.mean(),
        'std': importances.std()*importances.shape[0]**-0.5
    }, axis=1)
    return importances, scores.mean()


def get_single_feature_importance(features, clf, X, y, sample_weight, scoring, cv_splitter):
    """
    Reference implementation of single feature importance (SFI) concept.
    The model OOS CV score is computed for each feature separately and hence remove the substitution effect

    :param features: list of feature names
    :param clf: classifier model
    :param X: input features DataFrame
    :param scoring: scoring function name
    :param cv_splitter: CV splitter instance
    :return: DataFrame of mean and std of cv scores for each feature
    """
    importances = pd.DataFrame(columns=['mean', 'std'], dtype=float)
    for feature in features:
        scores = timeseries_cv_score(
           clf, X=X[[feature]], y=y, sample_weight=sample_weight, scoring=scoring, cv_splitter=cv_splitter
        )
        importances.loc[feature, 'mean'] = scores.mean()
        importances.loc[feature, 'std'] = scores.std()*scores.shape[0]**-0.5
    return importances
