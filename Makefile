.PHONY: install clean run-temperature run-air-quality run-all test data-clean

# Configuración
PYTHON = python
PIP = pip

# Instalar dependencias
install:
	$(PIP) install -r requirements.txt
	$(PYTHON) setup.py develop

env:
	pyenv install 3.9.18
	pyenv virtualenv 3.9.18 ts_weather
	@echo "Entorno virtual 'ts_weather' creado y activado."
	pyenv activate ts_weather

# Limpiar archivos generados
clean:
	find . -name "*.pyc" -exec rm -f {} \;
	find . -name "__pycache__" -exec rm -rf {} \;
	rm -rf models/*.h5
	rm -rf models/*.json
	rm -rf models_transformer/*.h5
	rm -rf models_transformer/*.json
	rm -rf data/*.csv
	rm -rf .pytest_cache
	rm -rf .openmeteo_cache

# Ejecutar para temperatura
run-temperature:
	$(PYTHON) main.py --target temperature_2m --model lstm

# Ejecutar para calidad del aire (PM2.5)
run-air-quality:
	$(PYTHON) main.py --target pm2_5 --model lstm

# Ejecutar para calidad del aire (PM10)
run-pm10:
	$(PYTHON) main.py --target pm10 --model lstm

# Ejecutar para calidad del aire (NO2)
run-no2:
	$(PYTHON) main.py --target nitrogen_dioxide --model lstm

# Ejecutar con Transformer para temperatura
run-temperature-transformer:
	$(PYTHON) main.py --target temperature_2m --model transformer

# Ejecutar con Transformer para calidad del aire
run-air-quality-transformer:
	$(PYTHON) main.py --target pm2_5 --model transformer

# Ejecutar todos los modelos (temperatura y calidad del aire)
run-all: run-temperature run-air-quality run-temperature-transformer run-air-quality-transformer

# Limpiar solo datos
data-clean:
	rm -rf data/*.csv
	rm -rf .openmeteo_cache

# Ayuda
help:
	@echo "Makefile para el proyecto de predicción de clima y calidad del aire"
	@echo ""
	@echo "Targets disponibles:"
	@echo "  install              Instalar dependencias"
	@echo "  clean                Limpiar archivos generados"
	@echo "  run-temperature      Ejecutar predicción de temperatura con LSTM"
	@echo "  run-air-quality      Ejecutar predicción de calidad del aire (PM2.5) con LSTM"
	@echo "  run-pm10             Ejecutar predicción de PM10 con LSTM"
	@echo "  run-no2              Ejecutar predicción de NO2 con LSTM"
	@echo "  run-temperature-transformer  Ejecutar predicción de temperatura con Transformer"
	@echo "  run-air-quality-transformer  Ejecutar predicción de calidad del aire con Transformer"
	@echo "  run-all              Ejecutar todos los modelos"
	@echo "  data-clean           Limpiar solo datos"
	@echo "  help                 Mostrar esta ayuda"