````markdown
#🌦️ TS Weather Pipeline  

**Pipeline reproducible para series temporales de meteorología y calidad del aire.**  
Proyecto desarrollado como trabajo para la materia *Series Temporales* — Maestría en Data Mining (UBA, 2025).  

---

## 📌 Resumen ejecutivo  
Repositorio que implementa un flujo completo —ingestión, procesamiento, ingeniería de características, modelado y evaluación— para pronóstico de variables meteorológicas (🌡️ temperatura) y de calidad del aire (💨 contaminantes) usando datos históricos.  

Se incluyen enfoques estadísticos clásicos (líneas base, ARIMA/VAR) y modelos de aprendizaje profundo (LSTM, Transformer), junto con notebooks reproducibles y scripts orquestadores.  

---

## 🎯 Objetivos  
1. Construir un pipeline modular y trazable para pronóstico univariado y multivariado de series temporales.  
2. Comparar desempeño entre modelos estadísticos y arquitecturas de deep learning.  
3. Garantizar reproducibilidad mediante entornos, checkpoints y cache de datos.  

---

## 📂 Contenido del repositorio  
- `config/` — parámetros y configuración del pipeline  
- `data/` — datos crudos y procesados  
- `models/` — modelos entrenados y checkpoints  
- `notebooks/` — notebooks con EDA, experimentos y visualizaciones  
- `ts_weather_pipeline/` — paquete con módulos del pipeline  
- `.openmeteo_cache.sqlite` — cache local de llamadas a Open-Meteo  
- `Makefile` — tareas automatizables (ejecución, limpieza, etc.)  
- `main.py` — orquestador / punto de entrada del pipeline  
- `requirements.txt` — dependencias Python reproducibles  
- `setup.py` — metadatos del paquete  
- `README.md` — este documento  

---

## ⚙️ Requisitos de entorno  
- Python 3.8+  
- Entorno virtual recomendado (venv / conda)  
- Dependencias listadas en `requirements.txt`  

---

## 🚀 Instalación  
```bash
git clone https://github.com/bitazaratustra/ts-weather-pipeline.git
cd ts-weather-pipeline

python -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\Activate.ps1   # Windows PowerShell

pip install -r requirements.txt
````

---

## ▶️ Ejecución rápida

* Pipeline completo:

  ```bash
  python main.py
  ```
* Abrir notebooks:

  ```bash
  jupyter lab
  ```
* Makefile (tareas disponibles):

  ```bash
  make help
  make run
  ```

---

## 🔄 Flujo del pipeline

1. Ingestión de datos históricos.
2. Preprocesamiento y limpieza.
3. Ingeniería de características (lags, ventanas, variables exógenas).
4. Modelado (ARIMA, VAR, LSTM, Transformer).
5. Evaluación con métricas (MAE, RMSE, R²).
6. Exportación de resultados y modelos.

---

## 📊 Resultados esperados

* Modelos entrenados y almacenados en `/models`.
* Pronósticos exportados en CSV y gráficas comparativas.
* Notebooks con análisis exploratorio y reportes.
* Métricas cuantitativas y visualizaciones de desempeño.

---

## 🗂️ Fuentes de datos

* **🌐 Open-Meteo** — series históricas meteorológicas vía API.
* **🏙️ BA Data** — calidad del aire en tiempo real, estaciones de la Ciudad Autónoma de Buenos Aires.
* **🌡️ SMN (Servicio Meteorológico Nacional)** — datos de temperatura medidos en estaciones oficiales de Capital Federal.

Estas fuentes aseguran representatividad y validez de los datos al provenir directamente de estaciones de medición locales.

---

## 📜 Reproducibilidad científica

* Dependencias controladas vía `requirements.txt`.
* Ejecución orquestada mediante `main.py`.
* Datos cacheados en `.openmeteo_cache.sqlite`.
* Notebooks documentados para replicar resultados.

---

## 📖 Licencia y citación

* Agregar archivo `LICENSE` con la licencia deseada (p. ej. MIT).
* Para citar este repositorio en trabajos académicos, se recomienda la referencia directa al código fuente en GitHub.

```
```
