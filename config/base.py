import os

CACHE_FILE = ".openmeteo_cache"
CACHE_EXPIRE = 3600  # seconds

BASE_HOURLY = [
    "temperature_2m", "relativehumidity_2m", "windspeed_10m", "precipitation"
]

AIR_QUALITY_HOURLY = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "aerosol_optical_depth"
]

def get_config(target_col, model_type="lstm"):
    base_config = {
        "lat": -34.6037,
        "lon": -58.3816,
        "start_date": "2010-01-01",
        "end_date": "2019-12-31",
        "target_col": target_col,
        "input_window": 48,
        "forecast_horizon": 24,
        "batch_size": 32,
        "epochs": 40,
        "learning_rate": 1e-3,
        "train_frac": 0.7,
        "val_frac": 0.15,
    }
    
    if model_type == "transformer":
        base_config.update({
            "patience": 6,
            "d_model": 128,
            "num_heads": 4,
            "ff_dim": 256,
            "num_transformer_blocks": 3,
            "model_dir": "models_transformer"
        })
    else:
        base_config.update({
            "lstm_units": 32,
            "model_dir": "models"
        })
    
    # Seleccionar variables según el objetivo
    if target_col in ["pm2_5", "pm10", "nitrogen_dioxide"]:
        base_config["hourly_vars"] = BASE_HOURLY + AIR_QUALITY_HOURLY
    else:
        base_config["hourly_vars"] = BASE_HOURLY
    
    return base_config