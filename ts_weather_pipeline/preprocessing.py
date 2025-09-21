import numpy as np
import pandas as pd
from scipy import signal

def fill_and_resample(df, freq="H"):
    idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(idx)
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].interpolate(limit_direction="both").ffill().bfill()
    return df

def add_time_features(df):
    df = df.copy()
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["month_sin"] = np.sin(2*np.pi*(df["month"]-1)/12)
    df["month_cos"] = np.cos(2*np.pi*(df["month"]-1)/12)
    return df

def add_lags(df, target_col, lags=(1,24,168)):
    df = df.copy()
    for l in lags:
        df[f"{target_col}_lag_{l}"] = df[target_col].shift(l)
    return df

def add_fourier(df, period_hours=24, n_harmonics=3, prefix="f24"):
    t = np.arange(len(df))
    for k in range(1, n_harmonics+1):
        df[f"{prefix}_sin_{k}"] = np.sin(2*np.pi*k*t/period_hours)
        df[f"{prefix}_cos_{k}"] = np.cos(2*np.pi*k*t/period_hours)
    return df

def lowpass(series, cutoff_hours=24, fs=1.0):
    nyq = 0.5*fs
    cutoff = 1.0/cutoff_hours
    normal_cutoff = cutoff/nyq
    b,a = signal.butter(4, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b,a,series)
