import pandas as pd

from data_structures.constants import EventCol


def drop_rare_labels(events: pd.DataFrame, min_pct=0.05, min_classes=2) -> pd.DataFrame:
    """
    Recursively drop labels with insufficient samples

    :param events: DataFrame with EventCol.LABEL column
    :param min_pct: minimum % of label value in the samples
    :param min_classes: minimum number of label classes
    :return: events DataFrame with rare labels removed
    """
    while True:
        df = events[EventCol.LABEL].value_counts(normalize=True)
        if df.min() > min_pct or df.shape[0] <= min_classes:
            break
        events = events[events[EventCol.LABEL] != df.index[df.argmin()]]
    return events
