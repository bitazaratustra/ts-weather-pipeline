import os, json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import callbacks

from .api import fetch_open_meteo_archive
from .preprocessing import fill_and_resample
from .features import add_time_feats, add_lags_and_rolls, add_fourier_features, lowpass_filter
from .analysis import adf_test, plot_decomposition, plot_fft
from .baselines import naive_forecast, fit_var, fit_arima
from .deep_learning import make_supervised, build_lstm_model
from .evaluation import evaluate_multi_horizon as evaluate_forecasts

def run_pipeline_example(target="temperature_2m", config=None):
    if config is None:
        from config import get_config
        config = get_config(target, "lstm")
    
    print("Downloading data...")
    df = fetch_open_meteo_archive(
        config["lat"], config["lon"], 
        config["start_date"], config["end_date"], 
        hourly_vars=config["hourly_vars"]
    )
    df = fill_and_resample(df, freq="H")
    print(f"Datos descargados: {df.shape}")
    print(f"Columnas disponibles: {df.columns.tolist()}")
    print(f"Rango temporal: {df.index.min()} to {df.index.max()}")

    print("ADF test (original):", adf_test(df[target]))
    plot_decomposition(df[target], period=24)
    plot_fft(df[target], fs=1.0)

    df = add_time_feats(df)
    df = add_lags_and_rolls(df, target_col=target)
    df = add_fourier_features(df, period_hours=24, n_harmonics=3)
    df[f"{target}_lowpass_24h"] = lowpass_filter(df[target].fillna(method="ffill").values)

    df = df.dropna()
    split_idx = int(len(df) * config["train_frac"])
    split_val = int(len(df) * (config["train_frac"] + config["val_frac"]))
    train_df, val_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:split_val], df.iloc[split_val:]

    naive_pred = naive_forecast(train_df[target], config["forecast_horizon"])
    true_next = test_df[target].values[:config["forecast_horizon"]]
    print("Naive eval:", evaluate_forecasts(true_next, naive_pred))

    try:
        var_res = fit_var(df[["temperature_2m", "relativehumidity_2m", "windspeed_10m"]])
        print("VAR fitted. AIC:", var_res.aic)
    except Exception as e:
        print(f"VAR failed: {str(e)}")
        import traceback
        traceback.print_exc()

    try:
        exog_feats = ["hour_sin","hour_cos"]
        arima_res = fit_arima(train_df[target], exog=train_df[exog_feats])
        arima_fc = arima_res.get_forecast(steps=config["forecast_horizon"], exog=test_df[exog_feats].iloc[:config["forecast_horizon"]])
        print("ARIMA eval:", evaluate_forecasts(test_df[target].iloc[:config["forecast_horizon"]].values, arima_fc.predicted_mean.values))
    except Exception as e:
        print("ARIMA failed:", e)

    scaler = MinMaxScaler()
    features = train_df.select_dtypes(include="number").columns.tolist()
    scaler.fit(train_df[features])
    train_scaled, val_scaled, test_scaled = [pd.DataFrame(scaler.transform(df[features]), columns=features, index=df.index) for df in [train_df,val_df,test_df]]

    input_width, output_width = config["input_window"], config["forecast_horizon"]
    X_train, y_train, _ = make_supervised(train_scaled, target, input_width, output_width)
    X_val, y_val, _ = make_supervised(val_scaled, target, input_width, output_width)
    X_test, y_test, _ = make_supervised(test_scaled, target, input_width, output_width)

    lstm = build_lstm_model((input_width, len(features)), output_width, config["lstm_units"], config["learning_rate"])
    es = callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    lstm.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=config["epochs"], batch_size=config["batch_size"], callbacks=[es], verbose=2)

    y_pred_lstm = lstm.predict(X_test, batch_size=config["batch_size"])
    eval_lstm = evaluate_forecasts(y_test, y_pred_lstm)
    print("LSTM eval:", eval_lstm)

    os.makedirs(config["model_dir"], exist_ok=True)
    lstm.save(f"{config['model_dir']}/lstm_{target}.h5")
    with open(f"{config['model_dir']}/scaler_{target}.json", "w") as fh: 
        json.dump({"features": features, "target": target}, fh)

    plt.plot(eval_lstm["mae_per_horizon"], label="LSTM MAE per horizon")
    plt.legend()
    plt.show()

    return eval_lstm