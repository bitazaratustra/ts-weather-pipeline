#!/usr/bin/env python3

"""
v13: Script de Análisis (Motor parametrizado)

v13: Añade "CNN_SIMPLE" (CNN apilable) y "CNN_RESNET" (ResNet)
     como opciones de modelo distintas. Ahora hay 5 arquitecturas.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import sys
import math
import argparse # Para leer argumentos

# --- 1. Configuration Parameters ---
# File and Data Params
FILE_NAME = 'weather-air-quality-clean.csv.bz2'
TARGET_POLLUTANTS = ['no2', 'pm10', 'co']
# Date Params
ANOMALY_START_DATE = '2020-03-20T00:00:00Z'
ANOMALY_END_DATE = '2020-05-10T23:00:00Z'
ANOMALY_PERIOD_1_END = '2020-04-12T23:00:00Z'
# Constantes de entrenamiento
VALIDATION_SPLIT = 0.2
# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- 2. Data Loading Function ---
def load_data(file_path):
    """Loads and performs initial parsing of the dataset."""
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path, compression='bz2')
    except FileNotFoundError:
        print(f"FATAL ERROR: File not found at '{file_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred loading the file: {e}")
        sys.exit(1)

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    print("Data loaded and date index set.")
    return df


# --- 3. Feature Engineering & Preprocessing Function ---
def preprocess_data(df, target_pollutants):
    """
    Creates target and feature columns.
    Targets: Median of all stations for each pollutant.
    Features: Median of primary weather stations, with OpenMeteo as fallback.
    """
    print("Starting preprocessing and feature engineering...")
    
    # 3.1. Create target columns using row-wise MEDIAN
    target_cols = []
    for pollutant in target_pollutants:
        pollutant_cols = [c for c in df.columns if f"{pollutant}_" in c]
        if not pollutant_cols:
            print(f"Warning: No columns found for target '{pollutant}'. Skipping.")
            continue
        
        target_col_name = f"{pollutant}_target"
        print(f"Combining {len(pollutant_cols)} columns for target: {pollutant_cols} -> {target_col_name}")
        df[target_col_name] = df[pollutant_cols].median(axis=1)
        target_cols.append(target_col_name)

    if not target_cols:
        print("FATAL ERROR: No target pollutant columns were successfully created.")
        sys.exit(1)
    print(f"\nCreated {len(target_cols)} target columns: {target_cols}")

    # 3.2. Define Weather Features (Covariates)
    base_weather_vars = ['temperature', 'relativehumidity', 'pressure', 'windspeed', 'windangle']
    primary_suffixes = ['_aeroparque', '_observatorio']
    fallback_suffix = '_openmeteo'
    features_list = []
    print("Combining primary weather stations (median) with openmeteo (fallback)...")
    
    for var in base_weather_vars:
        primary_cols = [f"{var}{s}" for s in primary_suffixes if f"{var}{s}" in df.columns]
        fallback_col = f"{var}{fallback_suffix}"
        combined_col_name = f"{var}_combined"
        
        if not primary_cols:
            if fallback_col in df.columns:
                print(f"  -> No primary stations for {var}. Using {fallback_col} directly.")
                df[combined_col_name] = df[fallback_col]
            else:
                print(f"Warning: No primary or fallback columns found for {var}.")
                continue
        else:
            print(f"  -> Combining {primary_cols} into {combined_col_name} (median)")
            df[combined_col_name] = df[primary_cols].median(axis=1)
        
        if fallback_col in df.columns:
            print(f"  -> Using {fallback_col} as fallback for {combined_col_name}")
            df[combined_col_name].fillna(df[fallback_col], inplace=True)
        else:
            print(f"Warning: No fallback column {fallback_col} found for {var}.")

        features_list.append(combined_col_name)

    # 3.3. Add Precipitation
    precip_col = 'precipitation_openmeteo'
    if precip_col in df.columns:
        print(f"Adding {precip_col} (has no other sources)")
        features_list.append(precip_col)
    
    if not features_list:
        print("FATAL ERROR: No weather features were successfully processed.")
        sys.exit(1)
    print(f"\nFinal features list ({len(features_list)}): {features_list}")

    # 3.4. Create Cyclical Time Features
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24.0)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    time_features = ['hour_sin', 'hour_cos', 'dayofyear_sin', 'dayofyear_cos']
    features_list.extend(time_features)
    
    # 3.5. Handle Missing Data (Interpolate)
    all_cols_to_process = target_cols + features_list
    print(f"\nInterpolating missing values (method='time') for all {len(all_cols_to_process)} target/feature columns...")
    df[all_cols_to_process] = df[all_cols_to_process].interpolate(method='time')
    
    # 3.6. Final DropNA
    print("Dropping any remaining NaN rows (e.g., from start of series)...")
    df.dropna(subset=all_cols_to_process, inplace=True)
    
    if df.empty:
        print("FATAL ERROR: Dataframe is empty after dropping NaNs.")
        sys.exit(1)
        
    print(f"Preprocessing complete. Final dataset shape: {df.shape}")
    return df, features_list, target_cols


# --- 4. Data Splitting and Scaling Function ---
def split_and_scale_data(df, features_list, target_cols, anomaly_start, anomaly_end):
    """
    Splits data into 'normal' (train/val) and 'anomaly' (test).
    Trains on data *around* the anomaly gap.
    """
    print("Splitting data into train/val (around gap) and anomaly (gap)...")
    
    anomaly_df = df.loc[anomaly_start : anomaly_end].copy()
    train_val_df_part1 = df.loc[df.index < anomaly_start].copy()
    train_val_df_part2 = df.loc[df.index > anomaly_end].copy()

    if train_val_df_part1.empty or train_val_df_part2.empty:
        print("FATAL ERROR: Not enough data before or after the anomaly period to train.")
        sys.exit(1)
    if anomaly_df.empty:
        print("FATAL ERROR: No data found for the anomaly period.")
        sys.exit(1)

    train_df_part1, val_df_part1 = train_test_split(train_val_df_part1, test_size=VALIDATION_SPLIT, shuffle=False)
    train_df_part2, val_df_part2 = train_test_split(train_val_df_part2, test_size=VALIDATION_SPLIT, shuffle=False)
    
    train_df = pd.concat([train_df_part1, train_df_part2])
    val_df = pd.concat([val_df_part1, val_df_part2])

    print(f"Training data:   {len(train_df)} rows (from 2 chunks)")
    print(f"Validation data: {len(val_df)} rows (from 2 chunks)")
    print(f"Anomaly (Test) data: {anomaly_df.index.min()} to {anomaly_df.index.max()} ({len(anomaly_df)} rows)")

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Fit scalers ONLY on training data
    feature_scaler.fit(train_df[features_list])
    target_scaler.fit(train_df[target_cols]) 

    # Transform all datasets
    for df_split in [train_df_part1, val_df_part1, train_df_part2, val_df_part2, anomaly_df]:
        df_split[features_list] = feature_scaler.transform(df_split[features_list])
        df_split[target_cols] = target_scaler.transform(df_split[target_cols])

    print("Data scaling complete.")
    
    return (train_df_part1, val_df_part1, 
            train_df_part2, val_df_part2, 
            anomaly_df, 
            feature_scaler, target_scaler)


# --- 5. Sequence Creation Function ---
def create_sequences(df, features_list, target_cols, seq_length):
    """Converts a time-series dataframe into X (sequences) and y (targets)"""
    X, y = [], []
    features = df[features_list].values
    target = df[target_cols].values # y is now a 2D array [N, n_targets]

    for i in range(len(df) - seq_length):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length])
        
    if not X:
        print("Warning: No sequences created. Data block may be too short for sequence length.")
        return np.array([]), np.array([])

    return np.array(X), np.array(y) 


# --- 6. PyTorch Model Definitions ---

# --- Model 1: CNN-LSTM ---
class CnnLstmModel(nn.Module):
    """Bidirectional CNN-LSTM Model with Dropout for Multi-Target Regression."""
    def __init__(self, num_features, num_targets, hidden_dim=64, cnn_out_channels=32, kernel_size=3, dropout_rate=0.3):
        super(CnnLstmModel, self).__init__()
        self.num_features = num_features
        
        self.cnn = nn.Conv1d(in_channels=num_features, out_channels=cnn_out_channels, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.cnn_dropout = nn.Dropout(dropout_rate)
        
        self.lstm = nn.LSTM(input_size=cnn_out_channels, hidden_size=hidden_dim, 
                            batch_first=True, bidirectional=True)
        self.lstm_dropout = nn.Dropout(dropout_rate)
        
        self.linear = nn.Linear(hidden_dim * 2, num_targets) 

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = self.relu(x)
        x = self.cnn_dropout(x)
        x = x.permute(0, 2, 1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        hidden = self.lstm_dropout(hidden)
        out = self.linear(hidden)
        return out

# --- Model 2: LSTM Only ---
class LstmOnlyModel(nn.Module):
    """Bidirectional LSTM-Only Model with Dropout."""
    def __init__(self, num_features, num_targets, hidden_dim=64, dropout_rate=0.3):
        super(LstmOnlyModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=num_features, 
            hidden_size=hidden_dim, 
            batch_first=True, 
            bidirectional=True
        )
        self.lstm_dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_dim * 2, num_targets)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        hidden = self.lstm_dropout(hidden)
        out = self.linear(hidden)
        return out

# --- Model 3: CNN Simple (NUEVO) ---
class CnnSimpleModel(nn.Module):
    """Un modelo 1D-CNN apilable."""
    def __init__(self, num_features, num_targets, channels, kernel_size, num_layers, dropout_rate):
        super(CnnSimpleModel, self).__init__()
        
        layers = []
        in_channels = num_features
        
        for i in range(num_layers):
            layers.append(nn.Conv1d(in_channels, channels, kernel_size, padding='same'))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_channels = channels # Las capas subsecuentes tienen 'channels' como entrada
            
        self.network = nn.Sequential(*layers)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(channels, num_targets)

    def forward(self, x):
        # Input x: (batch, seq_len, features)
        x = x.permute(0, 2, 1) # -> (batch, features, seq_len)
        x = self.network(x)     # -> (batch, channels, seq_len)
        x = self.pool(x)        # -> (batch, channels, 1)
        x = self.flatten(x)     # -> (batch, channels)
        out = self.linear(x)
        return out

# --- Model 4: CNN ResNet (RENOMBRADO) ---
class ResNetBlock(nn.Module):
    """Un bloque residual para 1D CNN."""
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class CnnResNetModel(nn.Module):
    """1D-CNN ResNet-style Model with Global Average Pooling."""
    def __init__(self, num_features, num_targets, channels, kernel_size, num_blocks, dropout_rate):
        super(CnnResNetModel, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv1d(num_features, channels, 3, padding='same'),
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )
        
        blocks = []
        for i in range(num_blocks):
            blocks.append(ResNetBlock(channels, channels, kernel_size, dropout_rate))
        self.blocks = nn.Sequential(*blocks)
        
        self.pool = nn.AdaptiveAvgPool1d(1) 
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(channels, num_targets)

    def forward(self, x):
        # Input x: (batch, seq_len, features)
        x = x.permute(0, 2, 1) # -> (batch, features, seq_len)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = self.flatten(x)
        out = self.linear(x)
        return out


# --- Model 5: Transformer ---
class PositionalEncoding(nn.Module):
    """Standard Positional Encoding for Transformers."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_buffer('pe', pe)

    def forward(self, x, seq_len):
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Transformer Encoder Model for Time Series."""
    def __init__(self, num_features, num_targets, d_model, nhead, num_encoder_layers, 
                 dim_feedforward, dropout, max_seq_len):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.input_embed = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_encoder_layers
        )
        self.linear = nn.Linear(d_model, num_targets)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.input_embed(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x, seq_len)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        out = self.linear(x)
        return out


# --- 7. Training and Evaluation Functions ---
def train_model(model, train_loader, val_loader, mse_criterion, mae_criterion, optimizer, 
                model_save_path, epochs, patience):
    """Main training loop. Tracks MSE, MAE, and RMSE."""
    print("Starting model training...")
    
    train_mse_history, val_mse_history = [], []
    train_mae_history, val_mae_history = [], []
    train_rmse_history, val_rmse_history = [], []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        train_mse_loss, train_mae_loss, train_rmse_loss = 0.0, 0.0, 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss_mse = mse_criterion(outputs, labels)
            loss_mse.backward()
            optimizer.step()
            
            train_mse_loss += loss_mse.item() * inputs.size(0)
            
            with torch.no_grad():
                loss_mae = mae_criterion(outputs, labels)
                train_mae_loss += loss_mae.item() * inputs.size(0)
                loss_rmse = torch.sqrt(loss_mse)
                train_rmse_loss += loss_rmse.item() * inputs.size(0)
            
        train_mse_loss /= len(train_loader.dataset)
        train_mae_loss /= len(train_loader.dataset)
        train_rmse_loss /= len(train_loader.dataset)
        train_mse_history.append(train_mse_loss)
        train_mae_history.append(train_mae_loss)
        train_rmse_history.append(train_rmse_loss)

        # --- Validation ---
        model.eval()
        val_mse_loss, val_mae_loss, val_rmse_loss = 0.0, 0.0, 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                
                loss_mse = mse_criterion(outputs, labels)
                val_mse_loss += loss_mse.item() * inputs.size(0)
                
                loss_mae = mae_criterion(outputs, labels)
                val_mae_loss += loss_mae.item() * inputs.size(0)
                
                loss_rmse = torch.sqrt(loss_mse)
                val_rmse_loss += loss_rmse.item() * inputs.size(0)

        val_mse_loss /= len(val_loader.dataset)
        val_mae_loss /= len(val_loader.dataset)
        val_rmse_loss /= len(val_loader.dataset)
        val_mse_history.append(val_mse_loss)
        val_mae_history.append(val_mae_loss)
        val_rmse_history.append(val_rmse_loss)
        
        print(f"Epoch {epoch:02d}/{epochs} | Train MSE: {train_mse_loss:.6f} | Val MSE: {val_mse_loss:.6f} | Val MAE: {val_mae_loss:.6f} | Val RMSE: {val_rmse_loss:.6f}")

        # --- Early Stopping Check (uses MSE) ---
        if val_mse_loss < best_val_loss:
            best_val_loss = val_mse_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"   (Validation MSE improved. Saving model to {model_save_path})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
                
    print("Training finished.")
    model.load_state_dict(torch.load(model_save_path))
    
    return (model, 
            train_mse_history, val_mse_history, 
            train_mae_history, val_mae_history,
            train_rmse_history, val_rmse_history)


# --- 8. Prediction Function ---
def get_anomaly_predictions(model, anomaly_df, features_list, target_cols, seq_length, scaler):
    """Generates predictions for the held-out anomaly period."""
    print("Generating predictions for the anomaly period...")
    model.eval()
    
    X_anomaly, y_anomaly = create_sequences(anomaly_df, features_list, target_cols, seq_length)
    results_index = anomaly_df.index[seq_length:]
    
    X_anomaly_tensor = torch.tensor(X_anomaly, dtype=torch.float32)
    y_anomaly_tensor = torch.tensor(y_anomaly, dtype=torch.float32)
    
    anomaly_dataset = TensorDataset(X_anomaly_tensor, y_anomaly_tensor)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=X_anomaly_tensor.size(0), shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for inputs, _ in anomaly_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
    predictions = np.concatenate(predictions) # Shape: [N, n_targets]
    
    # Inverse transform to get original scales
    y_all_unscaled = scaler.inverse_transform(y_anomaly)
    predictions_unscaled = scaler.inverse_transform(predictions)

    # Create Results DataFrame
    results_df = pd.DataFrame(index=results_index)
    
    pollutants = [c.replace('_target', '') for c in target_cols] 
    
    for i, poll in enumerate(pollutants):
        results_df[f'Actual_{poll}'] = y_all_unscaled[:, i]
        results_df[f'Predicted_{poll}'] = predictions_unscaled[:, i]
        results_df[f'Residual_{poll}'] = results_df[f'Actual_{poll}'] - results_df[f'Predicted_{poll}']
    
    print("Prediction generation complete.")
    return results_df, pollutants


# --- 9. Plotting Functions (Spanish Labels + SVG) ---

def plot_metric_curves(train_history, val_history, metric_name, file_name_base):
    """
    Generates and saves the plot of training and validation metrics.
    """
    print(f"Generating {metric_name} plot...")
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_history, label=f'{metric_name} de Entrenamiento', color='blue')
    plt.plot(val_history, label=f'{metric_name} de Validación', color='orange')
    plt.xlabel('Época')
    plt.ylabel(f"{metric_name} Agregado")
    plt.title(f'Curvas de Aprendizaje ({metric_name} Agregado)')
    plt.legend()
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    png_file = f'{file_name_base}.png'
    svg_file = f'{file_name_base}.svg'
    plt.savefig(png_file)
    plt.savefig(svg_file)
    print(f"Saved '{png_file}' and '{svg_file}'")


def plot_results(results_df, pollutants, period_1_end_date, model_type_name):
    """
    Generates and saves the final anomaly and residual plots
    for EACH pollutant.
    """
    print("Generating analysis plots for each pollutant...")
    period_1_end = pd.to_datetime(period_1_end_date)
    model_name_safe = model_type_name.lower()

    for poll in pollutants:
        print(f"  -> Plotting for {poll.upper()}...")
        actual_col = f'Actual_{poll}'
        predicted_col = f'Predicted_{poll}'
        residual_col = f'Residual_{poll}'
        
        if actual_col not in results_df.columns:
            print(f"    Skipping {poll.upper()} plots (no data).")
            continue
            
        plot_df = results_df[[actual_col, predicted_col, residual_col]].copy()
        poll_upper = poll.upper() # For titles

        # --- Plot 1: Actual vs. Predicted ---
        plt.figure(figsize=(15, 7))
        
        plt.plot(plot_df.index, plot_df[actual_col], label=f'{poll_upper} Real', color='blue', alpha=0.9)
        plt.plot(plot_df.index, plot_df[predicted_col], label=f'{poll_upper} Predicho (Contrafactual)', 
                 color='red', linestyle='--')
        plt.axvspan(plot_df.index.min(), period_1_end, color='orange', alpha=0.3, label='Período 1 (Estricto)')
        plt.axvspan(period_1_end, plot_df.index.max(), color='yellow', alpha=0.3, label='Período 2 (Extendido)')
        plt.title(f"{poll_upper} (Interpolación): Real vs. Predicho - {plot_df.index.min().date()} to {plot_df.index.max().date()}")
        plt.ylabel(f"Nivel de {poll_upper}")
        plt.xlabel("Fecha")
        plt.legend()
        
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        png_name_1 = f'{poll}_{model_name_safe}_anomaly_plot.png'
        svg_name_1 = f'{poll}_{model_name_safe}_anomaly_plot.svg'
        plt.savefig(png_name_1)
        plt.savefig(svg_name_1)
        print(f"Saved '{png_name_1}' and '{svg_name_1}'")

        # --- Plot 2: Residuals ---
        plt.figure(figsize=(15, 7))
        
        plt.plot(plot_df.index, plot_df[residual_col], label='Residual (Real - Predicho)', color='green')
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.axvspan(plot_df.index.min(), period_1_end, color='orange', alpha=0.3, label='Período 1 (Estricto)')
        plt.axvspan(period_1_end, plot_df.index.max(), color='yellow', alpha=0.3, label='Período 2 (Extendido)')
        plt.title(f"{poll_upper} Análisis de Residuales (Real - Predicho)")
        plt.ylabel(f"Nivel Residual de {poll_upper}")
        plt.xlabel("Fecha")
        plt.legend()
        
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        png_name_2 = f'{poll}_{model_name_safe}_residual_plot.png'
        svg_name_2 = f'{poll}_{model_name_safe}_residual_plot.svg'
        plt.savefig(png_name_2)
        plt.savefig(svg_name_2)
        print(f"Saved '{png_name_2}' and '{svg_name_2}'")
        
        # --- Print Summary Stats ---
        avg_residual_period_1 = plot_df.loc[:period_1_end, residual_col].mean()
        avg_residual_period_2 = plot_df.loc[period_1_end:, residual_col].mean()
        
        print(f"\n--- Anomaly Analysis Results ({poll_upper}) ---")
        print(f"Average Residual (Period 1): {avg_residual_period_1:.4f}")
        print(f"Average Residual (Period 2): {avg_residual_period_2:.4f}")
        print("(A negative residual means actual pollution was LOWER than predicted)\n")


# --- 10. Main Execution (MODIFICADO) ---
def main():
    """Main function to run the complete pipeline."""
    
    # --- NUEVO: Configuración de Argumentos ---
    parser = argparse.ArgumentParser(description='Run anomaly detection with a specific model.')
    
    # Argumentos de Modelo
    parser.add_argument('--model', type=str, required=True, 
                        choices=['CNN_SIMPLE', 'CNN_RESNET', 'LSTM', 'CNN_LSTM', 'TRANSFORMER'], 
                        help='Type of model architecture to run.')
    
    # Argumentos de Entrenamiento
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='L2 regularization (weight decay)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=72, help='Sequence length')
    
    # Argumentos de Arquitectura (Común)
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    
    # Argumentos de Arquitectura (CNN Simple)
    parser.add_argument('--cnn_simple_channels', type=int, default=64, help='Number of channels for Simple CNN')
    parser.add_argument('--cnn_simple_layers', type=int, default=3, help='Number of layers for Simple CNN')
    
    # Argumentos de Arquitectura (CNN ResNet)
    parser.add_argument('--cnn_resnet_channels', type=int, default=64, help='Number of channels for ResNet blocks')
    parser.add_argument('--cnn_resnet_blocks', type=int, default=3, help='Number of stacked ResNet blocks')

    # Argumento Común de Kernel (usado por ambos CNNs)
    parser.add_argument('--cnn_kernel_size', type=int, default=3, help='Kernel size for all CNNs')

    # Argumentos de Arquitectura (Transformer)
    parser.add_argument('--d_model', type=int, default=64, help='Transformer d_model (embedding dim)')
    parser.add_argument('--n_head', type=int, default=4, help='Transformer n_head (attention heads)')
    parser.add_argument('--num_encoder_layers', type=int, default=3, help='Transformer num_encoder_layers')
    parser.add_argument('--dim_feedforward', type=int, default=128, help='Transformer dim_feedforward')
    
    args = parser.parse_args()
    
    # Definir rutas de archivo basadas en los args
    MODEL_TYPE = args.model
    MODEL_SAVE_PATH = f'best_{MODEL_TYPE.lower()}_model.pth'
    
    print(f"--- Using device: {DEVICE} ---")
    print(f"--- Selected Model: {MODEL_TYPE} ---")
    print(f"--- Full Config: {args} ---")
    # --- FIN DE NUEVO ---
    
    # Step 1 & 2: Load and Preprocess
    df = load_data(FILE_NAME)
    df_processed, features, targets = preprocess_data(df, TARGET_POLLUTANTS)
    
    # Step 3: Split and Scale
    (train_p1, val_p1, train_p2, val_p2, anomaly_df, 
     f_scaler, t_scaler) = split_and_scale_data(
        df_processed, features, targets, ANOMALY_START_DATE, ANOMALY_END_DATE
    )

    # Step 4: Create Sequences (usa args.seq_len)
    print("Creating training and validation sequences...")
    X_train1, y_train1 = create_sequences(train_p1, features, targets, args.seq_len)
    X_train2, y_train2 = create_sequences(train_p2, features, targets, args.seq_len)
    X_val1, y_val1 = create_sequences(val_p1, features, targets, args.seq_len)
    X_val2, y_val2 = create_sequences(val_p2, features, targets, args.seq_len)
    
    if X_train1.size == 0 and X_train2.size == 0:
        print("FATAL ERROR: No training sequences created.")
        sys.exit(1)
    if X_val1.size == 0 and X_val2.size == 0:
        print("FATAL ERROR: No validation sequences created.")
        sys.exit(1)

    X_train = np.concatenate([X_train1, X_train2])
    y_train = np.concatenate([y_train1, y_train2])
    X_val = np.concatenate([X_val1, X_val2])
    y_val = np.concatenate([y_val1, y_val2])
    
    print(f"Total training sequences: {len(X_train)}")
    print(f"Total validation sequences: {len(X_val)}")
    
    # Step 5: Create PyTorch DataLoaders (usa args.batch_size)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                              torch.tensor(y_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Step 6: Initialize Model (usa args) ---
    num_features = len(features)
    num_targets = len(targets)
    
    print(f"\n--- Initializing model of type: {MODEL_TYPE} ---")
        
    if MODEL_TYPE == "CNN_SIMPLE":
        model = CnnSimpleModel(
            num_features=num_features,
            num_targets=num_targets,
            channels=args.cnn_simple_channels,
            kernel_size=args.cnn_kernel_size,
            num_layers=args.cnn_simple_layers,
            dropout_rate=args.dropout
        ).to(DEVICE)
        
    elif MODEL_TYPE == "CNN_RESNET":
        model = CnnResNetModel(
            num_features=num_features,
            num_targets=num_targets,
            channels=args.cnn_resnet_channels,
            kernel_size=args.cnn_kernel_size,
            num_blocks=args.cnn_resnet_blocks,
            dropout_rate=args.dropout
        ).to(DEVICE)
    
    elif MODEL_TYPE == "LSTM":
        model = LstmOnlyModel(
            num_features=num_features,
            num_targets=num_targets,
            dropout_rate=args.dropout
        ).to(DEVICE)
        
    elif MODEL_TYPE == "CNN_LSTM":
        model = CnnLstmModel(
            num_features=num_features, 
            num_targets=num_targets,
            dropout_rate=args.dropout
        ).to(DEVICE)
        
    elif MODEL_TYPE == "TRANSFORMER":
        model = TransformerModel(
            num_features=num_features,
            num_targets=num_targets,
            d_model=args.d_model,
            nhead=args.n_head,
            num_encoder_layers=args.num_encoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            max_seq_len=args.seq_len
        ).to(DEVICE)
    
    mse_criterion = nn.MSELoss() 
    mae_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Step 7: Train Model (usa args.epochs, args.patience)
    (trained_model, 
     train_mse_hist, val_mse_hist, 
     train_mae_hist, val_mae_hist,
     train_rmse_hist, val_rmse_hist) = train_model(
        model, train_loader, val_loader, mse_criterion, mae_criterion, optimizer, 
        MODEL_SAVE_PATH, args.epochs, args.patience
    )

    # Step 8: Get Anomaly Predictions (usa args.seq_len)
    results_df, pollutants_found = get_anomaly_predictions(
        trained_model, anomaly_df, features, targets, args.seq_len, t_scaler
    )

    # Step 9: Plot Results
    model_name_safe = MODEL_TYPE.lower()
    plot_base_mse = f"multitarget_{model_name_safe}_curvas_aprendizaje_mse"
    plot_base_mae = f"multitarget_{model_name_safe}_curvas_aprendizaje_mae"
    plot_base_rmse = f"multitarget_{model_name_safe}_curvas_aprendizaje_rmse"

    plot_metric_curves(train_mse_hist, val_mse_hist, "Loss (MSE)", plot_base_mse)
    plot_metric_curves(train_mae_hist, val_mae_hist, "Error Absoluto Medio (MAE)", plot_base_mae)
    plot_metric_curves(train_rmse_hist, val_rmse_hist, "Raíz del Error Cuadrático Medio (RMSE)", plot_base_rmse)
    
    plot_results(results_df, pollutants_found, ANOMALY_PERIOD_1_END, MODEL_TYPE)

    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    main()
