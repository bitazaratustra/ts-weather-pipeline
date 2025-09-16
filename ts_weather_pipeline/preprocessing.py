import pandas as pd

def fill_and_resample(df, freq="H", method="linear"):
    idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(idx)
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].interpolate(method=method).ffill().bfill()
    return df
