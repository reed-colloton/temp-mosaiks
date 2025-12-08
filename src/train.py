"""
Train ridge regression model to predict temperature from MOSAIKS features.

Usage:
    python train.py
    python train.py --test-size 0.3 --seed 123

Outputs:
    - Prints model performance metrics
    - Saves test predictions to output/test_predictions.csv
    - Saves model to output/model.joblib
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

from utils import c_to_f, calculate_metrics, print_metrics_header, print_metrics_row

# Paths
SRC_DIR = Path(__file__).parent
DATA_DIR = SRC_DIR.parent / "data"
OUTPUT_DIR = SRC_DIR / "output"
INPUT_FILE = DATA_DIR / "us_grid_025deg.csv"


def load_data(filepath=INPUT_FILE):
    """Load the prepared dataset."""
    print("Loading data...")
    data = pd.read_csv(filepath)
    print(f"Dataset: {len(data)} grid cells")
    return data


def prepare_features(data):
    """Extract features, target, and coordinates."""
    feature_cols = [f'X_{i}' for i in range(4000)]
    X = data[feature_cols].values
    y = data['avg_temp_c'].values  # Use Celsius for training
    lat = data['lat'].values
    lon = data['lon'].values
    return X, y, lat, lon


def train_test_split_data(X, y, lat, lon, test_size=0.2, random_state=42):
    """Split data into train and test sets using index-based splitting."""
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)

    return (X[train_idx], X[test_idx],
            y[train_idx], y[test_idx],
            lat[train_idx], lat[test_idx],
            lon[train_idx], lon[test_idx])


def train_baseline(lat_train, lat_test, y_train, y_test):
    """Train latitude-only baseline model."""
    model = LinearRegression()
    model.fit(lat_train.reshape(-1, 1), y_train)
    y_pred = model.predict(lat_test.reshape(-1, 1))
    metrics = calculate_metrics(y_test, y_pred)
    metrics['name'] = "Latitude Baseline"
    return metrics


def train_mosaiks(X_train, X_test, y_train, y_test):
    """Train MOSAIKS ridge regression model."""
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ridge regression with cross-validated alpha
    alphas = np.logspace(-4, 6, 50)
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_train_scaled, y_train)

    print(f"Best regularization alpha: {model.alpha_:.2e}")

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    return model, scaler, y_pred_train, y_pred_test


def main(test_size=0.2, random_state=42, data_path=None, output_dir=None):
    print("=" * 60)
    print("MOSAIKS Temperature Prediction - Training")
    print("=" * 60)

    # Use provided paths or defaults
    input_file = Path(data_path) if data_path else INPUT_FILE
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR

    # Create output directory if needed
    out_dir.mkdir(exist_ok=True)

    # Load data
    data = load_data(input_file)
    X, y, lat, lon = prepare_features(data)

    print(f"\nFeatures: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Temperature range: {y.min():.1f}C ({c_to_f(y.min()):.1f}F) to {y.max():.1f}C ({c_to_f(y.max()):.1f}F)")

    # Split data
    (X_train, X_test, y_train, y_test,
     lat_train, lat_test, lon_train, lon_test) = train_test_split_data(
         X, y, lat, lon, test_size=test_size, random_state=random_state
     )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Train baseline
    print("\n" + "-" * 40)
    print("Training baseline model (latitude only)...")
    baseline_metrics = train_baseline(lat_train, lat_test, y_train, y_test)

    # Train MOSAIKS model
    print("\n" + "-" * 40)
    print("Training MOSAIKS model...")
    model, scaler, y_pred_train, y_pred_test = train_mosaiks(
        X_train, X_test, y_train, y_test
    )

    train_metrics = calculate_metrics(y_train, y_pred_train)
    train_metrics['name'] = "MOSAIKS (Train)"
    test_metrics = calculate_metrics(y_test, y_pred_test)
    test_metrics['name'] = "MOSAIKS (Test)"

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()

    print_metrics_header()
    for m in [baseline_metrics, train_metrics, test_metrics]:
        print_metrics_row(m['name'], m['r2'], m['rmse'], m['mae'])

    improvement = test_metrics['r2'] - baseline_metrics['r2']
    print(f"\nMOSAIKS improvement over baseline: +{improvement*100:.1f}% R2")

    # Save test predictions
    predictions_df = pd.DataFrame({
        'lat': lat_test,
        'lon': lon_test,
        'actual_temp_c': y_test,
        'predicted_temp_c': y_pred_test,
        'error_c': y_test - y_pred_test,
        'actual_temp_f': c_to_f(y_test),
        'predicted_temp_f': c_to_f(y_pred_test),
        'error_f': (y_test - y_pred_test) * 9/5
    })
    pred_file = out_dir / "test_predictions.csv"
    predictions_df.to_csv(pred_file, index=False)
    print(f"\nSaved test predictions to {pred_file}")

    # Save model
    model_file = out_dir / "model.joblib"
    joblib.dump({'model': model, 'scaler': scaler}, model_file)
    print(f"Saved model to {model_file}")

    return model, scaler, test_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train MOSAIKS temperature prediction model'
    )
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data to use for testing (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to input CSV (default: data/us_grid_025deg.csv)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: src/output)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        test_size=args.test_size,
        random_state=args.seed,
        data_path=args.data_path,
        output_dir=args.output_dir
    )
