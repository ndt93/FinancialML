from datetime import datetime

import pandas as pd

df = pd.DataFrame({
    'Timestamp': pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-03']),
    'w1': [0.1, 0.2, 0.3],
    'w2': [-0.9, 0.8, 0.7],
}).set_index('Timestamp')

print(df.abs().sum(axis=1))
print(df.iloc[1, 1])