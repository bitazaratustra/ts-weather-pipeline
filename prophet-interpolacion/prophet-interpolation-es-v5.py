import matplotlib
# Force matplotlib to not use any Xwindows backend
matplotlib.use('Agg')

import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import matplotlib.pyplot as plt
import numpy as np
import os
import hashlib
import json
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------
# 1. CONFIGURATION (CONFIGURACIÓN)
# ---------------------------------------------------------
FILE_PATH = 'weather-air-quality-clean.csv.bz2'
CACHE_DIR = "model_cache"
OUTPUT_DIR = "analysis_output"

LOCKDOWN_START = '2020-03-20'
LOCKDOWN_END_FOR_MASKING = '2020-05-10'

TEST_PERIODS = [
    ('2020-03-20', '2020-04-12'),
    ('2020-03-20', '2020-05-10')
]

TARGET_GROUPS = {
    'pm10_median': ['pm10_centenario', 'pm10_cordoba', 'pm10_la_boca'],
    'no2_median':  ['no2_centenario', 'no2_cordoba', 'no2_la_boca'],
    'co_median':   ['co_centenario', 'co_cordoba', 'co_la_boca', 'co_palermo']
}

# Nombres para gráficos
DISPLAY_NAMES = {
    'pm10_median': 'PM10 (Mediana)',
    'no2_median':  'NO2 (Mediana)',
    'co_median':   'CO (Mediana)'
}

REGRESSOR_NAMES = [
    'reg_temperature', 'reg_relativehumidity', 'reg_pressure', 
    'reg_windspeed', 'reg_precipitation', 'reg_wind_sin', 'reg_wind_cos'
]

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS (Plotting in Spanish)
# ---------------------------------------------------------
def create_diagnostics_panel(target_key, results_df, train_indices, output_dir):
    """
    Generates a 6-panel diagnostic plot (Spanish). PNG ONLY (300 DPI).
    Includes Residuals vs Predicted and Standardized Residuals.
    """
    target_display = DISPLAY_NAMES.get(target_key, target_key)
    df_train = results_df.loc[train_indices].dropna(subset=['y', 'yhat'])
    
    if df_train.empty:
        return

    actuals = df_train['y']
    preds = df_train['yhat']
    residuals = actuals - preds
    
    # Calculate Standardized Residuals (Z-score)
    residuals_std = (residuals - residuals.mean()) / residuals.std()
    
    # Create 3 rows x 2 columns layout
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(f'Diagnóstico Avanzado: {target_display}', fontsize=16)
    
    # --- ROW 1: BASIC FIT ---
    # 1. Actual vs Predicted
    ax = axes[0, 0]
    ax.scatter(actuals, preds, alpha=0.3, s=10, color='blue')
    min_val = min(actuals.min(), preds.min())
    max_val = max(actuals.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ajuste Perfecto')
    ax.set_title('Observado vs. Predicho')
    ax.set_xlabel('Observación Real')
    ax.set_ylabel('Predicción del Modelo')
    ax.grid(True, alpha=0.3)

    # 2. Residuals over Time
    ax = axes[0, 1]
    ax.scatter(df_train['ds'], residuals, alpha=0.3, s=10, color='purple')
    ax.axhline(0, color='black', linestyle='--')
    ax.set_title('Residuos en el Tiempo (Estacionariedad)')
    ax.set_ylabel('Residuo (Real - Predicho)')
    ax.set_xlabel('Fecha')
    ax.grid(True, alpha=0.3)

    # --- ROW 2: HOMOSCEDASTICITY ---
    # 3. Residuals vs Predicted
    ax = axes[1, 0]
    ax.scatter(preds, residuals, alpha=0.3, s=10, color='teal')
    ax.axhline(0, color='black', linestyle='--')
    ax.set_title('Residuos vs. Predichos (Homocedasticidad)')
    ax.set_xlabel('Valor Predicho')
    ax.set_ylabel('Residuo')
    ax.grid(True, alpha=0.3)

    # 4. Standardized Residuals vs Predicted
    ax = axes[1, 1]
    ax.scatter(preds, residuals_std, alpha=0.3, s=10, color='darkorange')
    ax.axhline(0, color='black', linestyle='-')
    ax.axhline(2, color='red', linestyle='--', alpha=0.5, label='±2 SD')
    ax.axhline(-2, color='red', linestyle='--', alpha=0.5)
    ax.axhline(3, color='red', linestyle=':', alpha=0.5, label='±3 SD')
    ax.axhline(-3, color='red', linestyle=':', alpha=0.5)
    ax.set_title('Residuos Estandarizados vs. Predichos')
    ax.set_xlabel('Valor Predicho')
    ax.set_ylabel('Residuo Estandarizado (Z-Score)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- ROW 3: DISTRIBUTION ---
    # 5. Residual Histogram
    ax = axes[2, 0]
    ax.hist(residuals, bins=50, color='green', alpha=0.7, density=True)
    ax.set_title('Distribución de Residuos')
    ax.set_xlabel('Magnitud del Error')
    ax.set_ylabel('Frecuencia')
    ax.grid(True, alpha=0.3)
    
    # 6. Q-Q Plot
    ax = axes[2, 1]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Gráfico Q-Q (Normalidad)')
    ax.set_xlabel('Cuantiles Teóricos')
    ax.set_ylabel('Valores Ordenados')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save to PNG with 300 DPI
    base_path = os.path.join(output_dir, f"diagnostics_panel_{target_key}")
    plt.savefig(f"{base_path}.png", dpi=300)
    plt.close(fig)
    print(f"  -> Panel de diagnóstico avanzado guardado: {base_path}.png")

# ---------------------------------------------------------
# 3. LOAD DATA
# ---------------------------------------------------------
print("Cargando datos...")
try:
    df = pd.read_csv(FILE_PATH)
    df['ds'] = pd.to_datetime(df['date']).dt.tz_localize(None)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo {FILE_PATH}")
    exit(1)

# ---------------------------------------------------------
# 4. WEATHER REGRESSOR PREPARATION
# ---------------------------------------------------------
print("Procesando variables meteorológicas...")

# A. Linear Variables 
linear_recipes = [
    ('reg_temperature',      ['temperature_observatorio', 'temperature_aeroparque'],       'temperature_openmeteo'),
    ('reg_relativehumidity', ['relativehumidity_observatorio', 'relativehumidity_aeroparque'], 'relativehumidity_openmeteo'),
    ('reg_pressure',         ['pressure_observatorio', 'pressure_aeroparque'],             'pressure_openmeteo'),
    ('reg_windspeed',        ['windspeed_observatorio', 'windspeed_aeroparque'],           'windspeed_openmeteo'),
    ('reg_precipitation',    ['precipitation_observatorio', 'precipitation_aeroparque'],   'precipitation_openmeteo') 
]

for new_col, local_cols, fallback_col in linear_recipes:
    valid_locals = [c for c in local_cols if c in df.columns]
    if valid_locals:
        df[new_col] = df[valid_locals].mean(axis=1)
    else:
        df[new_col] = np.nan
    
    if fallback_col in df.columns:
        df[new_col] = df[new_col].fillna(df[fallback_col])
        
    df[new_col] = df[new_col].interpolate(method='linear').ffill().bfill()

# B. Circular Variables
wind_cols = ['windangle_observatorio', 'windangle_aeroparque', 'windangle_openmeteo']
for col in wind_cols:
    if col in df.columns:
        rads = df[col] * (np.pi / 180)
        df[f'{col}_sin'] = np.sin(rads)
        df[f'{col}_cos'] = np.cos(rads)

def fuse_components(suffix):
    locals_comp = [f'{c}_{suffix}' for c in ['windangle_observatorio', 'windangle_aeroparque'] if f'{c}_{suffix}' in df.columns]
    fallback_comp = f'windangle_openmeteo_{suffix}'
    if locals_comp:
        series = df[locals_comp].mean(axis=1)
    else:
        series = pd.Series(np.nan, index=df.index)
    if fallback_comp in df.columns:
        series = series.fillna(df[fallback_comp])
    return series.interpolate(method='linear').ffill().bfill()

df['reg_wind_sin'] = fuse_components('sin')
df['reg_wind_cos'] = fuse_components('cos')

# ---------------------------------------------------------
# 5. POLLUTION TARGET AGGREGATION
# ---------------------------------------------------------
print("Calculando medianas de contaminantes (Solo estaciones locales)...")
for target_name, source_cols in TARGET_GROUPS.items():
    valid_cols = [c for c in source_cols if c in df.columns]
    if valid_cols:
        df[target_name] = df[valid_cols].median(axis=1)

# ---------------------------------------------------------
# 6. ANALYSIS LOOP
# ---------------------------------------------------------
for target_col in TARGET_GROUPS.keys():
    if target_col not in df.columns: continue
    
    # Get pretty name for display
    target_display = DISPLAY_NAMES.get(target_col, target_col)

    if df[target_col].notna().sum() < 100:
        print(f"Saltando {target_display}: Datos insuficientes.")
        continue

    print(f"\n{'='*40}")
    print(f"Analizando Objetivo: {target_display}")
    print(f"{'='*40}")
    
    cols = ['ds', target_col] + REGRESSOR_NAMES
    data = df[cols].copy()
    data.rename(columns={target_col: 'y'}, inplace=True)
    
    # --- MASKING ---
    model_df = data.copy()
    mask_intervention = (model_df['ds'] >= LOCKDOWN_START) & (model_df['ds'] <= LOCKDOWN_END_FOR_MASKING)
    model_df.loc[mask_intervention, 'y'] = None 
    
    # --- CACHING ---
    params = {
        'daily': True, 'weekly': True, 'yearly': True, 'prior': 0.05,
        'regressors': sorted(REGRESSOR_NAMES),
        'target': target_col,
        'lockdown_start': LOCKDOWN_START,
        'lockdown_end': LOCKDOWN_END_FOR_MASKING
    }
    
    param_str = json.dumps(params, sort_keys=True)
    param_hash = hashlib.md5(param_str.encode("utf-8")).hexdigest()
    data_sig = f"{model_df['ds'].min()}-{model_df['ds'].max()}-{len(model_df)}-{model_df['y'].sum()}"
    data_hash = hashlib.md5(data_sig.encode("utf-8")).hexdigest()
    
    cache_key = f"{target_col}_{param_hash[:6]}_{data_hash[:6]}"
    cache_file = os.path.join(CACHE_DIR, f"prophet_{cache_key}.json")
    
    if os.path.exists(cache_file):
        print(f"Cache Hit! Cargando modelo desde {cache_file}...")
        with open(cache_file, 'r') as fin:
            m = model_from_json(json.load(fin))
    else:
        print(f"Cache Miss. Entrenando modelo...")
        m = Prophet(
            daily_seasonality=params['daily'], 
            weekly_seasonality=params['weekly'], 
            yearly_seasonality=params['yearly'],
            changepoint_prior_scale=params['prior']
        )
        for reg in REGRESSOR_NAMES:
            m.add_regressor(reg)
            
        m.fit(model_df)
        
        with open(cache_file, 'w') as fout:
            json.dump(model_to_json(m), fout)

    # -----------------------------------------------------
    # PREDICTION & DIAGNOSTICS
    # -----------------------------------------------------
    print("Generando predicciones contrafácticas...")
    forecast = m.predict(data)
    results = pd.merge(data[['ds', 'y']], forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')

    train_indices = model_df.dropna(subset=['y']).index
    y_true = model_df.loc[train_indices, 'y']
    y_pred = forecast.loc[train_indices, 'yhat']
    
    if len(y_true) > 0:
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        residuals = y_true - y_pred

        # Text Report
        report_path = os.path.join(OUTPUT_DIR, f"diagnostics_{target_col}.txt")
        with open(report_path, "w") as f:
            f.write(f"REPORTE DE DIAGNÓSTICO PARA {target_display}\n")
            f.write(f"====================================\n")
            f.write(f"R-Cuadrado (Varianza Explicada): {r2:.4f}\n")
            f.write(f"RMSE (Error Medio):              {rmse:.4f}\n")
            f.write(f"Sesgo Medio (Residual Mean):     {residuals.mean():.4f}\n")
        
        # B. Plots (Spanish, PNG only, 300 DPI)
        create_diagnostics_panel(target_col, results, train_indices, OUTPUT_DIR)
        
        # C. Components (PNG only, 300 DPI)
        fig_comp = m.plot_components(forecast)
        comp_base = os.path.join(OUTPUT_DIR, f"components_{target_col}")
        fig_comp.savefig(f"{comp_base}.png", dpi=300)
        plt.close(fig_comp)
        
        # D. [REMOVED] Standalone Residuals Histogram (Now in Panel)

    # -----------------------------------------------------
    # IMPACT PLOTTING (SPANISH)
    # -----------------------------------------------------
    plot_start = '2020-02-15'
    plot_end = '2020-06-15'
    mask_plot = (results['ds'] >= plot_start) & (results['ds'] <= plot_end)
    plot_data = results[mask_plot]
    
    if not plot_data.empty:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot translated labels
        ax.fill_between(plot_data['ds'], plot_data['yhat_lower'], plot_data['yhat_upper'], 
                         color='gray', alpha=0.3, label='Rango Esperado (Sin ASPO)')
        ax.plot(plot_data['ds'], plot_data['yhat'], color='#333333', linestyle='--', label='Media Esperada')
        
        ax.plot(plot_data['ds'], plot_data['y'], color='red', marker='o', markersize=3, 
                linestyle='-', linewidth=1, alpha=0.7, label='Observado Real (Mediana)')
        
        ax.axvspan(pd.to_datetime(LOCKDOWN_START), pd.to_datetime(LOCKDOWN_END_FOR_MASKING), 
                    color='green', alpha=0.15, label='ASPO (Aislamiento Estricto)')
        
        ax.set_title(f'Análisis de Impacto: {target_display}', fontsize=14)
        ax.set_ylabel('Concentración')
        ax.set_xlabel('Fecha')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Impact plots: PNG (300 DPI) + SVG
        base_name = os.path.join(OUTPUT_DIR, f"impact_{target_col}")
        plt.savefig(f"{base_name}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{base_name}.svg", bbox_inches='tight')
        print(f"  -> Gráficos de impacto guardados en {OUTPUT_DIR}")
        plt.close(fig)

    # -----------------------------------------------------
    # STATISTICAL OUTPUT
    # -----------------------------------------------------
    print("Calculando estadísticas de impacto...")
    for start_date, end_date in TEST_PERIODS:
        mask = (results['ds'] >= start_date) & (results['ds'] <= end_date)
        period_data = results[mask].dropna(subset=['y'])
        
        if len(period_data) == 0: continue

        mean_actual = period_data['y'].mean()
        mean_predicted = period_data['yhat'].mean()
        
        pct_change = 0
        if mean_predicted != 0:
            pct_change = ((mean_actual - mean_predicted) / mean_predicted) * 100
            
        print(f"  Periodo {start_date} al {end_date}:")
        print(f"    Esperado: {mean_predicted:.2f} | Observado: {mean_actual:.2f}")
        print(f"    Diferencia: {pct_change:.2f}%")

print("\nAnálisis Completado.")
