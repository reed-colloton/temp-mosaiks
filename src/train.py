"""
Train ridge regression model to predict temperature from MOSAIKS features.

Usage:
    python train.py

Outputs:
    - Prints model performance metrics
    - Saves test predictions to output/test_predictions.csv
    - Saves model to output/model.joblib
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Paths
SRC_DIR = Path(__file__).parent
OUTPUT_DIR = SRC_DIR / "output"
INPUT_FILE = SRC_DIR / "us_grid_025deg.csv"


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
    y = data['avg_temp_f'].values
    lat = data['lat'].values
    lon = data['lon'].values
    return X, y, lat, lon


def train_test_split_data(X, y, lat, lon, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    lat_train, lat_test = train_test_split(
        lat, test_size=test_size, random_state=random_state
    )
    lon_train, lon_test = train_test_split(
        lon, test_size=test_size, random_state=random_state
    )
    return (X_train, X_test, y_train, y_test,
            lat_train, lat_test, lon_train, lon_test)


def evaluate_model(y_true, y_pred, name="Model"):
    """Calculate and print evaluation metrics."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'name': name, 'r2': r2, 'rmse': rmse, 'mae': mae}


def train_baseline(lat_train, lat_test, y_train, y_test):
    """Train latitude-only baseline model."""
    model = LinearRegression()
    model.fit(lat_train.reshape(-1, 1), y_train)
    y_pred = model.predict(lat_test.reshape(-1, 1))
    return evaluate_model(y_test, y_pred, "Latitude Baseline")


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


def main():
    print("=" * 60)
    print("MOSAIKS Temperature Prediction - Training")
    print("=" * 60)

    # Create output directory if needed
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load data
    data = load_data()
    X, y, lat, lon = prepare_features(data)

    print(f"\nFeatures: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Temperature range: {y.min():.1f}F to {y.max():.1f}F")

    # Split data
    (X_train, X_test, y_train, y_test,
     lat_train, lat_test, lon_train, lon_test) = train_test_split_data(X, y, lat, lon)

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

    train_metrics = evaluate_model(y_train, y_pred_train, "MOSAIKS (Train)")
    test_metrics = evaluate_model(y_test, y_pred_test, "MOSAIKS (Test)")

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n{'Model':<25} {'R2':>10} {'RMSE':>10} {'MAE':>10}")
    print("-" * 55)

    for m in [baseline_metrics, train_metrics, test_metrics]:
        print(f"{m['name']:<25} {m['r2']:>10.4f} {m['rmse']:>9.2f}F {m['mae']:>9.2f}F")

    improvement = test_metrics['r2'] - baseline_metrics['r2']
    print(f"\nMOSAIKS improvement over baseline: +{improvement*100:.1f}% R2")

    # Save test predictions
    predictions_df = pd.DataFrame({
        'lat': lat_test,
        'lon': lon_test,
        'actual_temp_f': y_test,
        'predicted_temp_f': y_pred_test,
        'error_f': y_test - y_pred_test
    })
    pred_file = OUTPUT_DIR / "test_predictions.csv"
    predictions_df.to_csv(pred_file, index=False)
    print(f"\nSaved test predictions to {pred_file}")

    # Save model
    model_file = OUTPUT_DIR / "model.joblib"
    joblib.dump({'model': model, 'scaler': scaler}, model_file)
    print(f"Saved model to {model_file}")

    return model, scaler, test_metrics


if __name__ == "__main__":
    main()
