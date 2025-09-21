from setuptools import setup, find_packages

setup(
    name="ts_weather_pipeline",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "statsmodels",
        "scikit-learn",
        "tensorflow",
        "tqdm",
        "requests",
        "requests_cache"
    ],
)