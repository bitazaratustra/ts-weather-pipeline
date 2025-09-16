CACHE_FILE = ".openmeteo_cache"
CACHE_EXPIRE = 3600  

DEFAULT_HOURLY = ["temperature_2m", "relativehumidity_2m", "windspeed_10m", "precipitation"]

CONFIG = {
    "forecast_horizon_hours": 24,
    "input_window_hours": 168,
    "batch_size": 64,
    "epochs": 20,
    "learning_rate": 1e-3,
    "lstm_units": 128,
    "transformer_heads": 4,
    "transformer_dim": 128,
}
