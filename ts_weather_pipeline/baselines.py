import numpy as np
import math
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima.model import ARIMA

def naive_forecast(train, horizon):
    last = train.iloc[-1]
    return np.repeat(last, horizon)

def seasonal_naive(train, horizon, period=24):
    return train.iloc[-period: -period + horizon].values if horizon <= period else np.tile(train.iloc[-period:].values, math.ceil(horizon/period))[:horizon]

def fit_var(df, maxlags=24):
    model = VAR(df.dropna())
    return model.fit(maxlags=maxlags, ic="aic")

def fit_arima(series, order=(2,0,2), exog=None):
    model = ARIMA(series.dropna(), order=order, exog=exog)
    return model.fit()
