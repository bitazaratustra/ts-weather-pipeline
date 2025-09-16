from ts_weather_pipeline.pipeline import run_pipeline_example

if __name__ == "__main__":
    results = run_pipeline_example()
    print("Pipeline finished. Final LSTM eval:", results)
