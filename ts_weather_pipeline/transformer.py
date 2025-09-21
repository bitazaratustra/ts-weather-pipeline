import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import callbacks

from .api import fetch_open_meteo_archive
from .preprocessing import fill_and_resample, add_time_features, add_lags, add_fourier
from .analysis import adf_test, plot_decomposition, plot_fft, plot_pollutant_correlation
from .baselines import naive_forecast, fit_var, fit_arima
from .deep_learning import build_transformer_encoder
from .evaluation import evaluate_multi_horizon

def make_supervised_transformer(df, target_col, input_width, output_width):
    """
    Prepara datos para el modelo Transformer
    """
    data = df.select_dtypes(include="number").values
    n_samples = len(data) - input_width - output_width + 1
    X = np.zeros((n_samples, input_width, data.shape[1]))
    y = np.zeros((n_samples, output_width))
    
    for i in range(n_samples):
        X[i] = data[i:i+input_width]
        y[i] = data[i+input_width:i+input_width+output_width, df.columns.get_loc(target_col)]
    
    return X, y, df.columns.tolist()

def run_transformer_pipeline(target="pm2_5", config=None):
    """
    Pipeline para el modelo Transformer
    """
    if config is None:
        from config import get_config
        config = get_config(target, "transformer")
    
    print(f"Downloading data for {target}...")
    df = fetch_open_meteo_archive(
        config["lat"], config["lon"], 
        config["start_date"], config["end_date"], 
        hourly_vars=config["hourly_vars"]
    )
    df = fill_and_resample(df, freq="H")

    print("ADF test (original):", adf_test(df[target]))
    plot_decomposition(df[target], period=24)
    plot_fft(df[target], fs=1.0)
    
    # Si es calidad del aire, mostrar correlaciones
    if target in ["pm2_5", "pm10", "nitrogen_dioxide"]:
        pollutants = [c for c in ["pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide", "ozone"] if c in df.columns]
        plot_pollutant_correlation(df, pollutants)

    # Añadir características
    df = add_time_features(df)
    df = add_lags(df, target, lags=(1, 24, 168))
    df = add_fourier(df, period_hours=24, n_harmonics=3)
    
    df = df.dropna()
    
    # Dividir datos
    split_idx = int(len(df) * config["train_frac"])
    split_val = int(len(df) * (config["train_frac"] + config["val_frac"]))
    train_df, val_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:split_val], df.iloc[split_val:]
    
    # Escalar datos
    scaler = MinMaxScaler()
    features = train_df.select_dtypes(include="number").columns.tolist()
    scaler.fit(train_df[features])
    
    train_scaled = pd.DataFrame(scaler.transform(train_df[features]), columns=features, index=train_df.index)
    val_scaled = pd.DataFrame(scaler.transform(val_df[features]), columns=features, index=val_df.index)
    test_scaled = pd.DataFrame(scaler.transform(test_df[features]), columns=features, index=test_df.index)
    
    # Preparar datos supervisados
    input_width, output_width = config["input_window"], config["forecast_horizon"]
    X_train, y_train, _ = make_supervised_transformer(train_scaled, target, input_width, output_width)
    X_val, y_val, _ = make_supervised_transformer(val_scaled, target, input_width, output_width)
    X_test, y_test, _ = make_supervised_transformer(test_scaled, target, input_width, output_width)
    
    # Construir y entrenar modelo Transformer
    model = build_transformer_encoder(
        (input_width, len(features)), 
        output_width, 
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        ff_dim=config["ff_dim"],
        num_blocks=config["num_transformer_blocks"],
        dropout=0.1,
        lr=config["learning_rate"]
    )
    
    es = callbacks.EarlyStopping(monitor="val_loss", patience=config["patience"], restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=[es],
        verbose=2
    )
    
    # Evaluar modelo
    y_pred = model.predict(X_test, batch_size=config["batch_size"])
    eval_result = evaluate_multi_horizon(y_test, y_pred)
    print(f"Transformer evaluation for {target}:", eval_result)
    
    # Guardar modelo
    os.makedirs(config["model_dir"], exist_ok=True)
    model.save(f"{config['model_dir']}/transformer_{target}.h5")
    with open(f"{config['model_dir']}/scaler_{target}.json", "w") as f:
        json.dump({"features": features, "target": target}, f)
    
    # Graficar resultados
    plt.figure(figsize=(12, 6))
    plt.plot(eval_result["mae_per_horizon"], label="Transformer MAE per horizon")
    plt.xlabel("Horizon (hours)")
    plt.ylabel("MAE")
    plt.title(f"Transformer Performance for {target}")
    plt.legend()
    plt.savefig(f"{config['model_dir']}/transformer_{target}_performance.png")
    plt.show()
    
    return eval_result