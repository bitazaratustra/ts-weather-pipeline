import numpy as np
import pandas as pd
from scipy import signal

def add_time_feats(df):
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    return df

def add_lags_and_rolls(df, target_col="temperature_2m", lags=(1,24,168), windows=(24,168)):
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    for w in windows:
        df[f"{target_col}_roll_mean_{w}"] = df[target_col].rolling(window=w, min_periods=1).mean()
        df[f"{target_col}_roll_std_{w}"] = df[target_col].rolling(window=w, min_periods=1).std().fillna(0)
    return df

def add_fourier_features(df, period_hours=24, n_harmonics=3, prefix="hourly"):
    t = np.arange(len(df))
    for k in range(1, n_harmonics+1):
        df[f"{prefix}_sin_{k}"] = np.sin(2*np.pi*k*t/period_hours)
        df[f"{prefix}_cos_{k}"] = np.cos(2*np.pi*k*t/period_hours)
    return df

def lowpass_filter(series, cutoff_hours=24, fs=1.0):
    nyq = 0.5*fs
    cutoff = 1.0/cutoff_hours
    normal_cutoff = cutoff/nyq
    b, a = signal.butter(4, normal_cutoff, btype="low", analog=False)
    return signal.filtfilt(b, a, series)
