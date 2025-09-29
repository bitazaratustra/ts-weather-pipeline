Pipeline end-to-end para series temporales de clima y contaminación atmosférica, construído como proyecto para la materia de Series Temporales (Maestría Data Mining, UBA, 2025).

## Descripción general

Este repositorio implementa un flujo de trabajo completo para pronosticar variables meteorológicas (temperatura, etc.) y de calidad del aire (contaminación) mediante modelos de series temporales. Los pasos incluidos:

1. Descarga de datos históricos (API Open-Meteo u otras fuentes)  
2. Limpieza, preprocesamiento y *feature engineering*  
3. Análisis exploratorio  
4. Modelos base estadísticos (por ejemplo naïve, ARIMA, VAR)  
5. Modelos de deep learning (LSTM, Transformer, etc.)  
6. Evaluación y comparación de pronósticos  
7. Producción o entrega de resultados / visualizaciones  

El pipeline está organizado para fomentar modularidad y experimentación, permitiendo cambiar modelos, variables, horizontes de pronóstico, etc.

## Estructura del proyecto

Aquí una descripción de los directorios y archivos más relevantes:

├── config/ # Configuraciones (parámetros, rutas, constantes, etc.)

├── data/ # Datos crudos y procesados

├── models/ # Modelos entrenados, checkpoints

├── notebooks/ # Notebooks exploratorios y experimentales

├── ts_weather_pipeline/ # Código fuente del pipeline

│ ├── init.py

│ ├── data_ingestion.py

│ ├── preprocessing.py

│ ├── features.py

│ ├── modelling.py

│ ├── evaluation.py

│ └── utils.py

├── main.py # Punto de entrada al pipeline

├── requirements.txt # Dependencias de Python

├── setup.py # Instalación del paquete local (si aplica)

├── Makefile # Tareas automáticas (por ejemplo: correr pipeline completo)

└── .openmeteo_cache.sqlite # Cache local de llamadas a la API (opcional)


markdown
Copiar código

### Archivos clave

- **main.py**: script principal que ejecuta el pipeline en pasos (descarga → preparación → modelado → salida).  
- **requirements.txt**: lista de librerías necesarias.  
- **Makefile**: comandos útiles para automatizar tareas como limpiar datos, entrenar modelos, generar reportes, etc.  
- En `ts_weather_pipeline/`: cada módulo implementa una etapa (ingestión de datos, transformación, generación de features, definición de modelos, evaluación, utilidades auxiliares).

## Cómo empezar / instalación

1. Cloná este repositorio:

   ```bash
   git clone https://github.com/bitazaratustra/ts-weather-pipeline.git
   cd ts-weather-pipeline
   ```

```bash
# Copiar código
python3 -m venv venv
source venv/bin/activate     # en Linux / macOS
# o en Windows: venv\Scripts\activate
```
Instalá las dependencias:

```bash
Copiar código
pip install -r requirements.txt
```

