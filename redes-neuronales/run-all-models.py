#!/usr/bin/env python3

"""
Script de Ejecución (Runner) v5

Este script ejecuta 'analyze.py' para cada una de las
arquitecturas.

v5: Re-introduce la CNN simple como "CNN_SIMPLE" y renombra
    la ResNet a "CNN_RESNET", para un total de 5 arquitecturas.
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
        "epochs": 50,
        "patience": 15,
        "batch_size": 64,
        "seq_len": 72,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "dropout": 0.3,
        # --- Parámetros de CNN Simple ---
        "cnn_simple_channels": 64,  # Canales en cada capa
        "cnn_simple_layers": 8,     # <-- ¡Controla la profundidad!
        "cnn_kernel_size": 3,
    },
    {
        "model": "CNN_RESNET",
        "epochs": 50,
        "patience": 15,
        "batch_size": 64,
        "seq_len": 72,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "dropout": 0.2,
        # --- Parámetros de CNN ResNet ---
        "cnn_resnet_channels": 64,
        "cnn_kernel_size": 3,
        "cnn_resnet_blocks": 3
    },
    {
        "model": "LSTM",
        "epochs": 50,
        "patience": 15,
        "batch_size": 64,
        "seq_len": 72,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "dropout": 0.3,
    },
    {
        "model": "CNN_LSTM",
        "epochs": 50,
        "patience": 15,
        "batch_size": 64,
        "seq_len": 72,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "dropout": 0.3,
    },
    {
        "model": "TRANSFORMER",
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
        print(f"--- [ {datetime.now().strftime('%H:%M:%S')} ] INICIANDO: {arch} ---")
        
        # Construir el comando con todos los argumentos
        command = ['python', SCRIPT_TO_RUN]
        for key, value in config.items():
            command.append(f'--{key}')
            command.append(str(value))
        
        # Imprimir el comando que se va a ejecutar (para depuración)
        print(f"    Ejecutando: {' '.join(command)}")

        try:
            # Ejecutar el script y capturar la salida
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                encoding='utf-8', 
                check=True # Lanza una excepción si el script falla
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
            # Si el script falla (returncode != 0)
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
