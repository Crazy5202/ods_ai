import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

profile_df = pd.read_parquet("profile.parquet")

def clip_outlier(col):
    q01 = profile_df[col].quantile(0.01)
    q99 = profile_df[col].quantile(0.99)

    profile_df[col] = profile_df[col].clip(q01, q99)

profile_df['std_amount'] = profile_df['std_amount'].fillna(0)
profile_df['typical_timezone'] = profile_df['typical_timezone'].fillna(-999)

clip_cols = ['avg_amount', 'std_amount', 'std_amount', 'median_amount']
for col in clip_cols:
    clip_outlier(col)

profile_df.to_parquet('profile_clean.parquet')
'''
plt.hist(profile_df['typical_timezone'], bins=24)
plt.title('typical_timezone')
plt.grid(alpha=0.3)
plt.show()
'''
