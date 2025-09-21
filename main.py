import argparse
from ts_weather_pipeline.pipeline import run_pipeline_example
from ts_weather_pipeline.transformer import run_transformer_pipeline
from config.base import get_config

def main():
    parser = argparse.ArgumentParser(description='Predict weather or air quality')
    parser.add_argument('--target', type=str, default='temperature_2m', 
                       choices=['temperature_2m', 'pm2_5', 'pm10', 'nitrogen_dioxide'],
                       help='Target variable to predict')
    parser.add_argument('--model', type=str, default='lstm',
                       choices=['lstm', 'transformer'],
                       help='Model type to use')
    
    args = parser.parse_args()
    
    # Obtener configuración
    config = get_config(args.target, args.model)
    
    if args.model == "lstm":
        print(f"Running LSTM pipeline for {args.target}...")
        result = run_pipeline_example(target=args.target, config=config)
    else:
        print(f"Running Transformer pipeline for {args.target}...")
        result = run_transformer_pipeline(target=args.target, config=config)
    
    print(f"Results for {args.target}: {result}")

if __name__ == "__main__":
    main()