import argparse
import os
import glob
import sys
import pandas as pd
from datetime import datetime

try:
    from src.predict import PredictionEngine
except ImportError:
    print("Error: Could not import PredictionEngine from src.predict. Ensure you're running from the project root.")
    sys.exit(1)

MODELS_DIR = os.path.join(os.getcwd(), "models")


def find_latest_model_set(models_dir: str):
    """
    Find the latest saved model, scaler, and features files.
    Returns tuple: (model_path, scaler_path, features_path)
    """
    pattern_model = os.path.join(models_dir, "*_model.pkl")
    model_files = glob.glob(pattern_model)
    if not model_files:
        raise FileNotFoundError(f"No model files found in {models_dir}. Train a model first.")

    # Sort by modified time, newest first
    model_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    model_path = model_files[0]

    # Try to find matching scaler/features by stem prefix
    stem = os.path.basename(model_path).replace("_model.pkl", "")
    scaler_path = os.path.join(models_dir, f"{stem}_scaler.pkl")
    features_path = os.path.join(models_dir, f"{stem}_features.pkl")

    # Fallback: pick latest scaler/features if exact match missing
    if not os.path.exists(scaler_path):
        scalers = glob.glob(os.path.join(models_dir, "*_scaler.pkl"))
        scaler_path = sorted(scalers, key=lambda p: os.path.getmtime(p), reverse=True)[0] if scalers else None
    if not os.path.exists(features_path):
        features = glob.glob(os.path.join(models_dir, "*_features.pkl"))
        features_path = sorted(features, key=lambda p: os.path.getmtime(p), reverse=True)[0] if features else None

    return model_path, scaler_path, features_path


def main():
    parser = argparse.ArgumentParser(description="Batch predict network traffic from a CSV using the saved NIDS model.")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", default=None, help="Path to write predictions CSV (default: alongside input)")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to process from the input CSV")
    parser.add_argument("--delimiter", default=",", help="CSV delimiter (default ',')")

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    if not os.path.exists(MODELS_DIR):
        print(f"Error: Models directory not found: {MODELS_DIR}. Train a model first.")
        sys.exit(1)

    try:
        model_path, scaler_path, features_path = find_latest_model_set(MODELS_DIR)
    except Exception as e:
        print(f"Error locating model files: {e}")
        sys.exit(1)

    print("Loading prediction engine...")
    engine = PredictionEngine(model_path=model_path, scaler_path=scaler_path, feature_names_path=features_path)

    print(f"Reading input CSV: {input_path}")
    # Use pandas to optionally sample limit rows
    read_kwargs = {"delimiter": args.delimiter}
    try:
        df = pd.read_csv(input_path, **read_kwargs)
    except TypeError:
        # pandas uses 'sep' not 'delimiter' in older versions
        df = pd.read_csv(input_path, sep=args.delimiter)

    if args.limit is not None and args.limit > 0:
        df = df.head(args.limit)
        print(f"Limiting to first {len(df)} rows for processing.")

    # Run predictions via engine helper (handles feature selection and scaling)
    print("Running batch predictions...")
    results = engine.predict_from_csv(input_path)

    # If limit was provided, ensure output matches limited rows
    if args.limit is not None and args.limit > 0:
        results = results.head(args.limit)

    # Determine output path
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        base, ext = os.path.splitext(input_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{base}.predictions_{timestamp}.csv"

    print(f"Writing predictions to: {output_path}")
    results.to_csv(output_path, index=False)

    # Summary
    total = len(results)
    normal = int((results["Is_Attack"] == False).sum())
    attacks = int((results["Is_Attack"] == True).sum())
    avg_conf = float(results["Confidence"].mean()) if "Confidence" in results.columns else float("nan")

    print("\nSummary:")
    print(f"  Total samples: {total}")
    print(f"  Normal: {normal}")
    print(f"  Attacks: {attacks}")
    print(f"  Avg confidence: {avg_conf:.4f}")


if __name__ == "__main__":
    main()
