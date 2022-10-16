import numpy as np
import pandas as pd


def get_mean_decrease_impurity(model, feature_names: list[str]):
    """
    Get the in-sample impurity decrease contributed by each feature in tree-based model.
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
