import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

from financial_ml.evaluation.cross_validation import timeseries_cv_score, PurgedKFold
from financial_ml.explain.feature_importance import (
    get_mean_decrease_impurity,
    get_mean_decrease_accuracy,
    get_single_feature_importance
)


# TODO: Tests taking too long. To simplify
def get_test_data(n_features=40, n_informative=10, n_redundant=10, n_samples=10000):
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=n_informative, n_redundant=n_redundant, random_state=0,
        shuffle=False
    )
    obs_times = pd.bdate_range(periods=n_samples,
                               freq=pd.tseries.offsets.BDay(),
                               end=pd.datetime.today())
    X = pd.DataFrame(X, index=obs_times)
    y = pd.Series(y, index=obs_times)
    features = ['I_' + str(i) for i in range(n_informative)] + ['R_' + str(i) for i in range(n_redundant)]
    features += ['N_' + str(i) for i in range(n_features - len(features))]
    X.columns = features
    sample_weight = pd.Series(1./y.shape[0], index=y.index)
    event_times = pd.Series(y.index, index=y.index)
    return X, y, sample_weight, event_times


def get_feature_importance(
        X, y, sample_weight, event_times=None, n_estimators=1000, cv=10,
        max_samples=1., num_threads=8, embargo_pct=0, scoring='accuracy', method='SFI',
):
    n_jobs = -1 if num_threads > 1 else 1
    clf = DecisionTreeClassifier(criterion='entropy', max_features=1, class_weight='balanced')
    clf = BaggingClassifier(
        base_estimator=clf, n_estimators=n_estimators, max_features=1.,
        max_samples=max_samples, oob_score=True, n_jobs=n_jobs
    )
    fit = clf.fit(X=X, y=y, sample_weight=sample_weight.values)
    oob = fit.oob_score_
    if method == 'MDI':
        imp = get_mean_decrease_impurity(fit, feature_names=X.columns)
        oos = timeseries_cv_score(
            clf, X=X, y=y, cv=cv, sample_weight=sample_weight,
            event_times=event_times, embargo_pct=embargo_pct, scoring=scoring
        ).mean()
    elif method == 'MDA':
        imp, oos = get_mean_decrease_accuracy(
            clf, X=X, y=y, cv=cv, sample_weight=sample_weight,
            event_times=event_times, embargo_pct=embargo_pct, scoring=scoring
        )
    elif method == 'SFI':
        cv_splitter = PurgedKFold(n_splits=cv, event_times=event_times, embargo_pct=embargo_pct)
        oos = timeseries_cv_score(
            clf, X=X, y=y, sample_weight=sample_weight, scoring=scoring, cv_splitter=cv_splitter
        ).mean()
        imp = get_single_feature_importance(X.columns, clf, X, y, sample_weight, scoring, cv_splitter)
    else:
        raise NotImplementedError(method)
    return imp, oob, oos


def test_mdi_feature_importance():
    X, y, sample_weight, event_times = get_test_data()
    imp, oob, oos = get_feature_importance(X, y, sample_weight, event_times=event_times, method='MDI')
    np.testing.assert_array_equal(np.sort(imp.index.values), np.sort(X.columns.values))
    np.testing.assert_array_equal(imp.columns, ['mean', 'std'])
    assert oob > 0.9
    assert oos > 0.8


def test_mda_feature_importance():
    X, y, sample_weight, event_times = get_test_data()
    imp, oob, oos = get_feature_importance(X, y, sample_weight, event_times=event_times, method='MDA')
    np.testing.assert_array_equal(np.sort(imp.index.values), np.sort(X.columns.values))
    np.testing.assert_array_equal(imp.columns, ['mean', 'std'])
    assert oob > 0.9
    assert oos > 0.8


def test_sfi_feature_importance():
    X, y, sample_weight, event_times = get_test_data()
    imp, oob, oos = get_feature_importance(X, y, sample_weight, cv=3, event_times=event_times, method='SFI')
    np.testing.assert_array_equal(np.sort(imp.index.values), np.sort(X.columns.values))
    np.testing.assert_array_equal(imp.columns, ['mean', 'std'])
