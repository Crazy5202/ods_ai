import pandas as pd

# перевод в другой тип данных
def to_type(df, dtype, cols):
    df[cols] = df[cols].astype(dtype)
    return df

# вытаскиваем час и день недели из даты
def to_datetime(df):
    df['event_dttm'] = pd.to_datetime(df['event_dttm'])

    df['event_hour'] = df['event_dttm'].dt.hour.astype('int8')
    df['event_dow'] = df['event_dttm'].dt.dayofweek.astype('int8')

    return df.drop(columns=['event_dttm'])

# словарь из типичных действий клиента
def typical_customer(df_pretrain: pd.DataFrame) -> dict:
    profile = {}

    def safe_mode(series):
        if series.empty:
            return None
        mode_vals = series.mode()
        return mode_vals.iloc[0] if not mode_vals.empty else None

    grouped = df_pretrain.groupby('customer_id', sort=False)

    for customer, customer_ops in grouped:
        amounts = customer_ops['operaton_amt'].dropna()

        mcc_series = customer_ops['mcc_code'].dropna()
        mcc_frequency = mcc_series.value_counts(normalize=True)

        profile[customer] = {
            # деньги
            'avg_amount': amounts.mean() if not amounts.empty else 0,
            'std_amount': amounts.std() if not amounts.empty else 0,
            'median_amount': amounts.median() if not amounts.empty else 0,
            'max_normal_amount': amounts.quantile(0.95) if not amounts.empty else 0, #95% операций до максимума
            'min_amount': amounts.min() if not amounts.empty else 0,
            'avg_operations_per_day': len(customer_ops) / 365,
            'total_operation': len(customer_ops),

            # день недели и час
            'typical_hours': customer_ops['event_hour'].mode().tolist() if not customer_ops.empty else [],
            'typical_days': customer_ops['event_day'].mode().tolist() if not customer_ops.empty else [],

            # данные о канале, ос и часовому поясу
            'typical_chanel': safe_mode(customer_ops['channel_indicator_type']),
            'typical_os': safe_mode(customer_ops['operating_system_type']),
            'typical_timezone': safe_mode(customer_ops['timezone']),

            # самые популярные категории покупок
            'top_mcc': mcc_frequency.head(5).to_dict(),
        }
    return profile

df_pretrain_1 = pd.read_parquet("pretrain_part_1.parquet")
df_pretrain_2 = pd.read_parquet("pretrain_part_2.parquet")
df_pretrain_3 = pd.read_parquet("pretrain_part_3.parquet")

use_cols = [
    'customer_id',
    'operaton_amt',
    'event_dttm',
    'mcc_code',
    'channel_indicator_type',
    'operating_system_type',
    'timezone',
]

drop_cols = [x for x in df_pretrain_1.columns if x not in use_cols]
df_pretrain_1 = df_pretrain_1.drop(columns=drop_cols)
df_pretrain_2 = df_pretrain_2.drop(columns=drop_cols)
df_pretrain_3 = df_pretrain_3.drop(columns=drop_cols)

df_pretrain_1 = to_datetime(df_pretrain_1)
df_pretrain_2 = to_datetime(df_pretrain_2)
df_pretrain_3 = to_datetime(df_pretrain_3)

cols_float = ['operaton_amt', 'timezone', 'operating_system_type']
df_pretrain_1 = to_type(df_pretrain_1, 'float32', cols_float)
df_pretrain_2 = to_type(df_pretrain_2, 'float32', cols_float)
df_pretrain_3 = to_type(df_pretrain_3, 'float32', cols_float)

df_pretrain = pd.concat([df_pretrain_1, df_pretrain_2, df_pretrain_3])

profile = typical_customer(df_pretrain)

df_prof_all = pd.DataFrame.from_dict(profile, orient='index').reset_index().rename(columns={'index': 'customer_id'})

df_prof_all.to_parquet("profile.parquet", index=False)