#!/usr/bin/env python3

"""
Script de Ejecución (Runner) v8

Este script ejecuta 'analyze.py' para cada una de las
arquitecturas.

v8: Añade la bandera 'validation_strategy' para elegir entre
    "interleaved" (bloques rotativos) y "simple" (80/20).
"""

import subprocess
import os
import sys
from datetime import datetime

# --- Configuración ---
SCRIPT_TO_RUN = 'analyze.py'
LOG_FILE = 'model_benchmark_log.txt'

# --- CONFIGURACIÓN DE HIPERPARÁMETROS ---
# Aquí es donde defines TODOS los parámetros para cada modelo.

MODEL_CONFIGS = [
    {
        "model": "CNN_SIMPLE",
        "validation_strategy": "simple", # <-- NUEVA BANDERA: "interleaved" o "simple"
        "plot_fit": True,
        "epochs": 300,
        "patience": 150,
        "batch_size": 64,
        "seq_len": 72,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "dropout": 0.3,
        "cnn_simple_channels": 64,
        "cnn_simple_layers": 3,
        "cnn_kernel_size": 3,
        "train_split_days": 240, # Usado solo si validation_strategy="interleaved"
        "val_split_days": 60,    # Usado solo si validation_strategy="interleaved"
    },
    {
        "model": "CNN_RESNET",
        "validation_strategy": "simple",
        "plot_fit": True,
        "epochs": 50,
        "patience": 15,
        "batch_size": 64,
        "seq_len": 72,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "dropout": 0.2,
        "cnn_resnet_channels": 64,
        "cnn_kernel_size": 3,
        "cnn_resnet_blocks": 3,
        "train_split_days": 240,
        "val_split_days": 60,
    },
    {
        "model": "LSTM",
        "validation_strategy": "simple",
        "plot_fit": True,
        "epochs": 100,
        "patience": 15,
        "batch_size": 64,
        "seq_len": 72,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "dropout": 0.3,
        "train_split_days": 240,
        "val_split_days": 60,
    },
    {
        "model": "CNN_LSTM",
        "validation_strategy": "simple",
        "plot_fit": True,
        "epochs": 100,
        "patience": 15,
        "batch_size": 64,
        "seq_len": 72,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "dropout": 0.3,
        "train_split_days": 240,
        "val_split_days": 60,
    },
    {
        "model": "TRANSFORMER",
        "validation_strategy": "simple",
        "plot_fit": True,
        "epochs": 50,
        "patience": 7,
        "batch_size": 32,
        "seq_len": 72,
        "lr": 3e-5,
        "weight_decay": 1e-6,
        "dropout": 0.1,
        "d_model": 128,
        "n_head": 8,
        "num_encoder_layers": 4,
        "dim_feedforward": 256,
        "train_split_days": 240,
        "val_split_days": 60,
    }
]

# -----------------------------------------------

def main():
    print(f"Iniciando el benchmark de {len(MODEL_CONFIGS)} configuraciones...")
    print(f"El registro se guardará en: {LOG_FILE}\n")
    
    # Limpiar el archivo de registro al inicio
    try:
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f"Inicio del Benchmark: {datetime.now()}\n")
            f.write("="*80 + "\n")
    except IOError as e:
        print(f"Error: No se puede escribir en el archivo de registro '{LOG_FILE}'. {e}")
        sys.exit(1)

    # Iterar sobre cada configuración de modelo
    for config in MODEL_CONFIGS:
        arch = config['model']
        print(f"--- [ {datetime.now().strftime('%H:%M:%S')} ] INICIANDO: {arch} (Estrategia: {config.get('validation_strategy', 'simple')}) ---")
        
        # Construir el comando con todos los argumentos
        command = ['python', SCRIPT_TO_RUN]
        for key, value in config.items():
            if isinstance(value, bool):
                if value:
                    command.append(f'--{key}')
            else:
                command.append(f'--{key}')
                command.append(str(value))
        
        print(f"    Ejecutando: {' '.join(command)}")

        try:
            # Ejecutar el script y capturar la salida
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                encoding='utf-8', 
                check=True
            )
            
            # Si tiene éxito, escribir stdout en el log
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(f"\n--- INICIO DEL REGISTRO: {arch} ---\n")
                f.write(f"Configuración: {config}\n\n")
                f.write(result.stdout)
                f.write(f"\n--- FIN DEL REGISTRO: {arch} ---\n")
                f.write("="*80 + "\n")
                
            print(f"--- [ {datetime.now().strftime('%H:%M:%S')} ] COMPLETADO: {arch} ---")

        except subprocess.CalledProcessError as e:
            # Si el script falla
            print(f"--- [ {datetime.now().strftime('%H:%M:%S')} ] ¡ERROR! FALLÓ: {arch} ---")
            print(f"    Ver {LOG_FILE} para detalles del error.")
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(f"\n--- ¡¡¡ERROR!!! FALLÓ: {arch} ---\n\n")
                f.write(f"Configuración: {config}\n\n")
                f.write("--- STDOUT (si existe) ---\n")
                f.write(e.stdout)
                f.write("\n--- STDERR ---\n")
                f.write(e.stderr)
                f.write(f"\n--- FIN DEL ERROR: {arch} ---\n")
                f.write("="*80 + "\n")
                
        except FileNotFoundError:
            print(f"Error: 'python' no se encuentra. Asegúrese de que Python esté en su PATH.")
            sys.exit(1)

    print(f"\nBenchmark completado. Todos los resultados guardados en {LOG_FILE}.")

if __name__ == "__main__":
    main()
