import os, json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import callbacks

from .config import CONFIG, DEFAULT_HOURLY
from .api import fetch_open_meteo_archive
from .preprocessing import fill_and_resample
from .features import add_time_feats, add_lags_and_rolls, add_fourier_features, lowpass_filter
from .analysis import adf_test, plot_decomposition, plot_fft
from .baselines import naive_forecast, fit_var, fit_arima
from .deep_learning import make_supervised, build_lstm_model
from .evaluation import evaluate_forecasts

def run_pipeline_example(lat=-34.6037, lon=-58.3816, start="2010-01-01", end="2019-12-31", target="temperature_2m"):
    print("Downloading data...")
    df = fetch_open_meteo_archive(lat, lon, start, end, hourly_vars=DEFAULT_HOURLY)
    df = fill_and_resample(df, freq="H")

    print("ADF test (original):", adf_test(df[target]))
    plot_decomposition(df[target], period=24)
    plot_fft(df[target], fs=1.0)

    df = add_time_feats(df)
    df = add_lags_and_rolls(df, target_col=target)
    df = add_fourier_features(df, period_hours=24, n_harmonics=3)
    df[f"{target}_lowpass_24h"] = lowpass_filter(df[target].fillna(method="ffill").values)

    df = df.dropna()
    split_test = int(len(df)*0.8)
    split_val = int(len(df)*0.9)
    train_df, val_df, test_df = df.iloc[:split_test], df.iloc[split_test:split_val], df.iloc[split_val:]

    naive_pred = naive_forecast(train_df[target], CONFIG["forecast_horizon_hours"])
    true_next = test_df[target].values[:CONFIG["forecast_horizon_hours"]]
    print("Naive eval:", evaluate_forecasts(true_next, naive_pred))

    try:
        var_res = fit_var(df[["temperature_2m", "relativehumidity_2m", "windspeed_10m"]])
        print("VAR fitted. AIC:", var_res.aic)
    except Exception as e:
        print("VAR failed:", e)

    try:
        exog_feats = ["hour_sin","hour_cos"]
        arima_res = fit_arima(train_df[target], exog=train_df[exog_feats])
        arima_fc = arima_res.get_forecast(steps=CONFIG["forecast_horizon_hours"], exog=test_df[exog_feats].iloc[:CONFIG["forecast_horizon_hours"]])
        print("ARIMA eval:", evaluate_forecasts(test_df[target].iloc[:CONFIG["forecast_horizon_hours"]].values, arima_fc.predicted_mean.values))
    except Exception as e:
        print("ARIMA failed:", e)

    scaler = MinMaxScaler()
    features = train_df.select_dtypes(include="number").columns.tolist()
    scaler.fit(train_df[features])
    train_scaled, val_scaled, test_scaled = [pd.DataFrame(scaler.transform(df[features]), columns=features, index=df.index) for df in [train_df,val_df,test_df]]

    input_width, output_width = CONFIG["input_window_hours"], CONFIG["forecast_horizon_hours"]
    X_train,y_train,_ = make_supervised(train_scaled, target, input_width, output_width)
    X_val,y_val,_ = make_supervised(val_scaled, target, input_width, output_width)
    X_test,y_test,_ = make_supervised(test_scaled, target, input_width, output_width)

    lstm = build_lstm_model((input_width,len(features)), output_width, CONFIG["lstm_units"], CONFIG["learning_rate"])
    es = callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    lstm.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=CONFIG["epochs"],batch_size=CONFIG["batch_size"],callbacks=[es],verbose=2)

    y_pred_lstm = lstm.predict(X_test, batch_size=CONFIG["batch_size"])
    eval_lstm = evaluate_forecasts(y_test, y_pred_lstm)
    print("LSTM eval:", eval_lstm)

    os.makedirs("models", exist_ok=True)
    lstm.save("models/lstm_model.h5")
    with open("models/scaler_and_feats.json","w") as fh: json.dump({"features": features}, fh)

    plt.plot(eval_lstm["mae_per_horizon"], label="LSTM MAE per horizon")
    plt.legend(); plt.show()

    return eval_lstm
