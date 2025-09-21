import numpy as np

def evaluate_multi_horizon(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2, axis=0))
    mape = np.mean(np.abs((y_true - y_pred)/np.where(y_true==0,1e-8,y_true)), axis=0)*100
    return {"mae_per_horizon": mae, "rmse_per_horizon": rmse, "mape_per_horizon": mape,
            "mae": mae.mean(), "rmse": rmse.mean(), "mape": mape.mean()}

# Alias para compatibilidad
evaluate_forecasts = evaluate_multi_horizon