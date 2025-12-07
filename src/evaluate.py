"""
Evaluate different model configurations and generate comparison report.

Compares:
- Latitude only baseline
- Latitude + Longitude
- MOSAIKS features only
- MOSAIKS + Latitude
- MOSAIKS + Latitude + Longitude
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Paths
SRC_DIR = Path(__file__).parent
OUTPUT_DIR = SRC_DIR / "output"
INPUT_FILE = SRC_DIR / "us_grid_025deg.csv"


def main():
    print("=" * 60)
    print("MOSAIKS Temperature Prediction - Model Comparison")
    print("=" * 60)

    # Create output directory if needed
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load data
    data = pd.read_csv(INPUT_FILE)
    print(f"\nDataset: {len(data)} grid cells")

    # Prepare features
    feature_cols = [f'X_{i}' for i in range(4000)]
    X = data[feature_cols].values
    y = data['avg_temp_c'].values  # Use Celsius for training
    lat = data['lat'].values
    lon = data['lon'].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    lat_train, lat_test = train_test_split(lat, test_size=0.2, random_state=42)
    lon_train, lon_test = train_test_split(lon, test_size=0.2, random_state=42)

    # Standardize MOSAIKS features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Standardize coordinates
    lat_train_s = (lat_train - lat_train.mean()) / lat_train.std()
    lat_test_s = (lat_test - lat_train.mean()) / lat_train.std()
    lon_train_s = (lon_train - lon_train.mean()) / lon_train.std()
    lon_test_s = (lon_test - lon_train.mean()) / lon_train.std()

    results = []
    alphas = np.logspace(-4, 6, 50)

    # 1. Latitude only
    print("\nTraining: Latitude only...")
    model = LinearRegression()
    model.fit(lat_train.reshape(-1, 1), y_train)
    pred = model.predict(lat_test.reshape(-1, 1))
    results.append({
        'model': 'Latitude only',
        'r2': r2_score(y_test, pred),
        'rmse': np.sqrt(mean_squared_error(y_test, pred)),
        'mae': mean_absolute_error(y_test, pred)
    })

    # 2. Latitude + Longitude
    print("Training: Latitude + Longitude...")
    geo_train = np.column_stack([lat_train, lon_train])
    geo_test = np.column_stack([lat_test, lon_test])
    model = LinearRegression()
    model.fit(geo_train, y_train)
    pred = model.predict(geo_test)
    results.append({
        'model': 'Lat + Lon',
        'r2': r2_score(y_test, pred),
        'rmse': np.sqrt(mean_squared_error(y_test, pred)),
        'mae': mean_absolute_error(y_test, pred)
    })

    # 3. MOSAIKS only
    print("Training: MOSAIKS only...")
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    mosaiks_alpha = model.alpha_
    results.append({
        'model': 'MOSAIKS only',
        'r2': r2_score(y_test, pred),
        'rmse': np.sqrt(mean_squared_error(y_test, pred)),
        'mae': mean_absolute_error(y_test, pred)
    })

    # 4. MOSAIKS + Latitude
    print("Training: MOSAIKS + Latitude...")
    X_lat_train = np.column_stack([X_train_s, lat_train_s])
    X_lat_test = np.column_stack([X_test_s, lat_test_s])
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_lat_train, y_train)
    pred = model.predict(X_lat_test)
    results.append({
        'model': 'MOSAIKS + Lat',
        'r2': r2_score(y_test, pred),
        'rmse': np.sqrt(mean_squared_error(y_test, pred)),
        'mae': mean_absolute_error(y_test, pred)
    })

    # 5. MOSAIKS + Latitude + Longitude
    print("Training: MOSAIKS + Lat + Lon...")
    X_geo_train = np.column_stack([X_train_s, lat_train_s, lon_train_s])
    X_geo_test = np.column_stack([X_test_s, lat_test_s, lon_test_s])
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_geo_train, y_train)
    pred = model.predict(X_geo_test)
    results.append({
        'model': 'MOSAIKS + Lat + Lon',
        'r2': r2_score(y_test, pred),
        'rmse': np.sqrt(mean_squared_error(y_test, pred)),
        'mae': mean_absolute_error(y_test, pred)
    })

    # Print results
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)

    print(f"\n{'Model':<25} {'R2':>10} {'RMSE (C)':>10} {'RMSE (F)':>10} {'MAE (C)':>10} {'MAE (F)':>10}")
    print("-" * 75)

    for r in results:
        rmse_f = r['rmse'] * 9/5  # Convert C to F scale
        mae_f = r['mae'] * 9/5
        print(f"{r['model']:<25} {r['r2']:>10.4f} {r['rmse']:>10.2f} {rmse_f:>10.2f} {r['mae']:>10.2f} {mae_f:>10.2f}")

    # Summary
    baseline_r2 = results[0]['r2']
    mosaiks_r2 = results[2]['r2']
    improvement = mosaiks_r2 - baseline_r2

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nBaseline (Latitude only):  R2 = {baseline_r2:.4f}")
    print(f"Best (MOSAIKS only):       R2 = {mosaiks_r2:.4f}")
    print(f"Improvement:               +{improvement*100:.1f}% R2")
    print(f"\nBest regularization alpha: {mosaiks_alpha:.2e}")

    # Save results
    results_df = pd.DataFrame(results)
    output_file = OUTPUT_DIR / "model_comparison.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}")

    return results


if __name__ == "__main__":
    main()
