import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from scipy.fft import rfft, rfftfreq

def adf_test(series):
    series = series.dropna()
    stat, pvalue, usedlag, nobs, *_ = adfuller(series, autolag="AIC")
    return {"stat": stat, "pvalue": pvalue, "usedlag": usedlag, "nobs": nobs}

def plot_decomposition(series, period=24):
    stl = STL(series.dropna(), period=period, robust=True)
    res = stl.fit()
    res.plot()
    plt.show()
    return res

def plot_fft(series, fs=1.0):
    x = series.dropna().values
    N = len(x)
    yf = rfft(x - x.mean())
    xf = rfftfreq(N, 1/fs)
    plt.plot(xf, np.abs(yf))
    plt.xlabel("Freq (1/hour)")
    plt.ylabel("Amplitude")
    plt.title("FFT")
    plt.show()
