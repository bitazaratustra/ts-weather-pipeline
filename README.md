````markdown
# TS Weather Pipeline

Pipeline end-to-end para series temporales de clima y calidad del aire (pronóstico de temperatura y contaminación) — proyecto de pruebas/experimentos para la materia *Series Temporales* (Maestría Data Mining, UBA).

> Este README está pensado para pegar directamente en `README.md`. Contiene instrucciones rápidas para instalar, ejecutar y entender la estructura del repo.

---

## ¿Qué hace este repositorio?

- Descarga datos meteorológicos históricos (uso de Open-Meteo / cache local).  
- Limpieza y preprocesamiento de series temporales (lags, ventanas, imputación).  
- Generación de *features* y pipelines reproducibles.  
- Modelos baseline estadísticos (naïve, ARIMA/VAR) y modelos de deep learning (LSTM / Transformer) para pronosticar temperatura y contaminantes.  
- Evaluación de forecasts (MAE, RMSE, visualizaciones) y guardado de checkpoints/resultados.  
- Notebooks con experimentos y ejemplos reproducibles.

---

## Requisitos

- Python 3.8+ (recomendado 3.9/3.10)  
- Git  
- Recursos opcionales: GPU para entrenar modelos deep learning más rápido

Las dependencias principales están en `requirements.txt`. Instálalas en un entorno virtual:

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1

pip install -r requirements.txt
````

---

## Estructura del repositorio (extracto)

```
.
├── config/                  # archivos de configuración / parámetros del pipeline
├── data/                    # datos crudos y procesados
├── models/                  # modelos entrenados / checkpoints / pesos
├── notebooks/               # notebooks de exploración y experimentos
├── ts_weather_pipeline/     # paquete principal: ingestion, preprocessing, features, modelling, evaluation
├── .openmeteo_cache.sqlite  # cache local de llamadas a Open-Meteo (si aplica)
├── main.py                  # script / orquestador principal del pipeline
├── requirements.txt
├── Makefile                 # tareas útiles (run, clean, train, etc.)
└── README.md
```

> Nota: adaptá la ruta/archivo de configuración dentro de `config/` según tu entorno (p.ej. cambiar ciudad, rango de fechas, horizonte de predicción).

---

## Configuración rápida

1. Cloná el repo:

```bash
git clone https://github.com/bitazaratustra/ts-weather-pipeline.git
cd ts-weather-pipeline
```

2. Activá el entorno e instalá dependencias (ver sección anterior).

3. Revisá y editá (si corresponde) los parámetros en `config/`:

   * fechas inicial/final
   * ubicación / coordenadas
   * horizonte de predicción (n pasos)
   * nombre del sensor / variable objetivo (p. ej. `temperature`, `pm2_5`)

4. (Opcional) Si requieres ajustar cache o permisiones: borrá ` .openmeteo_cache.sqlite` para forzar re-descarga.

---

## Ejecutar el pipeline

> El repo contiene un `main.py` que sirve como punto de entrada. Su comportamiento por defecto es orquestar las etapas principales: ingestión → preprocesamiento → entrenamiento → evaluación.

Comando rápido (pipeline completo):

```bash
python main.py
```

Consejos útiles:

* Para ver opciones/help (si `main.py` implementa argumentos CLI):

```bash
python main.py --help
```

* Si preferís ejecutar paso a paso (recomendado mientras probás):

  1. Descargar / actualizar datos (ingest)
  2. Ejecutar preprocesamiento / features
  3. Entrenar modelo
  4. Evaluar y generar reportes

(En muchos proyectos similares eso se hace con flags como `--step ingest` / `--step train`, o con funciones dentro de `ts_weather_pipeline/*.py`. Revisá `main.py` para los flags exactos.)

---

## Notebooks

* Abrí `notebooks/` para ver experimentos reproducibles (EDA, comparación de modelos, tuning).
* Para abrir Jupyter:

```bash
jupyter lab   # o jupyter notebook
```

* Si querés ejecutar un notebook desde terminal:

```bash
jupyter nbconvert --to notebook --execute notebooks/tu_notebook.ipynb --output notebooks/out/tu_notebook_executed.ipynb
```

---

## Salidas esperadas

Al ejecutar el pipeline/ver notebooks deberías obtener:

* Modelos entrenados en `models/` (archivos/Checkpoints).
* CSVs con pronósticos en `data/` o `results/` (según configuración).
* Gráficas comparando pronóstico vs real (generadas por notebooks o scripts de evaluación).
* Logs de entrenamiento / métricas (MAE, RMSE, R² según lo implementado).

---

## Buenas prácticas y recomendaciones

* Usá *time-based split* para train/val/test (no mezclá aleatoriamente).
* Guardá scaler/transformaciones junto con modelos (para reproducibilidad).
* Evitá *data leakage*: al crear features con ventanas/lags, respetá el orden temporal.
* Documentá cada experimento: parámetros, seed, rango de fechas y archivos de entrada.
* Versioná modelos (fecha + métricas) para compararlos fácilmente.

---

## Extensiones sugeridas

* Añadir variables exógenas (tráfico, eventos, emisiones industriales).
* Hacer forecasts probabilísticos (intervalos de confianza).
* Desplegar modelo como API (FastAPI / Flask) para servir pronósticos.
* Pipeline orquestado con Airflow/Prefect para ejecución periódica y monitoreo.

---

## Problemas comunes

* Si falla la descarga de datos: verificá tu conexión y que las coordenadas/fechas sean válidas.
* Si modelos no convergen: reducí el LR, aumentá regularización o simplificá la arquitectura (menos capas).
* Si ves discrepancias entre notebooks y scripts: asegurate de usar las mismas versiones de dependencias y el mismo `requirements.txt`.

---

## Licencia y contacto

* Agregá la licencia que prefieras (p. ej. `MIT`) creando un archivo `LICENSE`.
* Si querés que adapte este README para que incluya comandos específicos (por ejemplo los flags exactos de `main.py`, nombres de notebooks o una lista detallada de dependencias), pegá aquí el contenido de `main.py` y lo ajusto y lo dejo listo para copiar/pegar.

---

¡Listo! Este README está preparado para pegarlo directamente en `README.md`. Si querés que lo personalice aún más (p. ej. con ejemplos de salida reales, flags exactos de `main.py` o una lista completa de notebooks), pegame los archivos clave y lo dejo perfecto para tu repo.

```
::contentReference[oaicite:0]{index=0}
```
