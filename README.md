````markdown
# TS Weather Pipeline

**Pipeline reproducible para series temporales de meteorología y calidad del aire.**  
Proyecto desarrollado como trabajo para la materia *Series Temporales* — Maestría en Data Mining (Facultad de Ciencias Exactas y Naturales, UBA, 2025).

---

## Resumen ejecutivo
Repositorio que implementa un flujo completo —ingestión, procesamiento, ingeniería de características, modelado y evaluación— para pronóstico de variables meteorológicas (p. ej. temperatura) y de calidad del aire (p. ej. PM₂.₅) usando datos históricos de Open-Meteo. El proyecto combina enfoques estadísticos clásicos (líneas base, ARIMA/VAR) y modelos de aprendizaje profundo (LSTM, Transformer) en notebooks reproducibles y scripts orquestadores.

---

## Objetivos
1. Construir un pipeline modular y trazable para pronóstico univariado y multivariado de series temporales meteorológicas.  
2. Comparar desempeño entre bases estadísticas y arquitecturas recurrentes / transformacionales.  
3. Garantizar reproducibilidad mediante entornos, checkpoints y cache de datos.

---

## Contenido del repositorio
- `config/` — parámetros y configuración del pipeline.  
- `data/` — datos crudos y procesados.  
- `models/` — modelos entrenados y checkpoints.  
- `notebooks/` — notebooks con EDA, experimentos y visualizaciones.  
- `ts_weather_pipeline/` — paquete con módulos del pipeline.  
- `.openmeteo_cache.sqlite` — cache local de llamadas a Open-Meteo.  
- `Makefile` — tareas automatizables (ejecución, limpieza, etc.).  
- `main.py` — orquestador / punto de entrada del pipeline.  
- `requirements.txt` — dependencias Python reproducibles.  
- `setup.py` — metadatos del paquete.  
- `README.md` — este documento.

---

## Requisitos de entorno
- Python 3.8+  
- Recomendado: entorno virtual (venv / conda)  
- Dependencias listadas en `requirements.txt`

---

## Instalación (procedimiento reproducible)
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/bitazaratustra/ts-weather-pipeline.git
   cd ts-weather-pipeline
````

2. Crear y activar un entorno virtual:

   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux / macOS
   # venv\Scripts\Activate.ps1   # Windows PowerShell
   ```
3. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```

---

## Ejecución rápida

* Pipeline completo (orquestador):

  ```bash
  python main.py
  ```
* Abrir y ejecutar notebooks:

  ```bash
  jupyter lab
  # o
  jupyter notebook
  ```
* Makefile (tareas disponibles según el repositorio):

  ```bash
  make help
  make run
  ```

---

## Flujo del pipeline (alto nivel)

1. **Ingestión de datos**: descarga de series históricas desde Open-Meteo y almacenamiento en `data/` con cache en `.openmeteo_cache.sqlite`.
2. **Preprocesamiento**: limpieza temporal, imputación de faltantes y alineamiento de frecuencias.
3. **Ingeniería de características**: creación de lags, ventanas deslizantes, variables calendario y exógenas.
4. **Modelado**: comparación entre líneas base (naïve), modelos ARIMA/VAR y modelos secuenciales (LSTM) o basados en atención (Transformer).
5. **Evaluación**: métricas de error (MAE, RMSE, R²) y visualización de predicciones vs. observaciones.
6. **Persistencia**: exportación de checkpoints y pronósticos a `models/` y `data/results/`.

---

## Metodología y diseño experimental

* Se emplea *time-aware splitting* (train / validation / test) respetando el orden temporal.
* Para cada experimento se versionan: parámetros, seed, lista de features y fecha/hora de ejecución.
* Resultados cuantitativos (MAE / RMSE) y cualitativos (plots temporales) se registran en los notebooks de `notebooks/`.

---

## Datos

* Fuente primaria: **Open-Meteo** (descarga programática).
* Cache: `.openmeteo_cache.sqlite` para evitar múltiples llamadas y asegurar reproducibilidad de la data adquirida.

---

## Salidas esperadas

* Modelos entrenados y sus pesos en `models/`.
* Pronósticos exportados en CSV en `data/` o `results/`.
* Notebooks ejecutables con gráficos de comparación y tablas de métricas.

---

## Reproducibilidad científica

* Control de dependencias vía `requirements.txt`.
* Script orquestador único: `main.py` para reproducir el flujo completo.
* Notebooks documentados para replicar análisis y gráficos.

---

Fuentes: repositorio y archivos del proyecto en GitHub. :contentReference[oaicite:0]{index=0}
```
