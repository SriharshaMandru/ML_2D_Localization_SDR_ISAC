import pandas as pd


def numeric_feature_columns(df, drop_cols):
    return [
        c
        for c in df.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]


def distance_feature_columns(df):
    keywords = [
        "rssi",
        "power",
        "amp",
        "corr",
        "frequency_mhz",
        "txrx",
    ]

    return [
        c
        for c in df.columns
        if any(k in c.lower() for k in keywords)
        and pd.api.types.is_numeric_dtype(df[c])
    ]
