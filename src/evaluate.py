"""
Evaluate different model configurations and generate comparison report.

Compares:
- Latitude only baseline
- Latitude + Longitude
- MOSAIKS features only
- MOSAIKS + Latitude
- MOSAIKS + Latitude + Longitude

Also includes spatial cross-validation experiments to assess generalization.

Usage:
    python evaluate.py
    python evaluate.py --test-size 0.3 --seed 123
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

from utils import calculate_metrics, print_metrics_header, print_metrics_row

# Paths
SRC_DIR = Path(__file__).parent
DATA_DIR = SRC_DIR.parent / "data"
OUTPUT_DIR = SRC_DIR / "output"
INPUT_FILE = DATA_DIR / "us_grid_025deg.csv"


def train_and_evaluate(X_train, X_test, y_train, y_test, model_name, use_ridge=True, alphas=None):
    """Train a model and return metrics."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if use_ridge:
        if alphas is None:
            alphas = np.logspace(-4, 6, 50)
        model = RidgeCV(alphas=alphas, cv=5)
    else:
        model = LinearRegression()

    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    metrics = calculate_metrics(y_test, pred)
    metrics['model'] = model_name

    return metrics, model


def run_model_comparison(X, y, lat, lon, train_idx, test_idx, alphas):
    """Run all model configurations on given train/test split."""
    results = []

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    lat_train, lat_test = lat[train_idx], lat[test_idx]
    lon_train, lon_test = lon[train_idx], lon[test_idx]

    # Standardize MOSAIKS features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Standardize coordinates using sklearn for consistency
    coord_scaler = StandardScaler()
    coords_train = np.column_stack([lat_train, lon_train])
    coords_test = np.column_stack([lat_test, lon_test])
    coords_train_s = coord_scaler.fit_transform(coords_train)
    coords_test_s = coord_scaler.transform(coords_test)
    lat_train_s, lon_train_s = coords_train_s[:, 0], coords_train_s[:, 1]
    lat_test_s, lon_test_s = coords_test_s[:, 0], coords_test_s[:, 1]

    # 1. Latitude only
    model = LinearRegression()
    model.fit(lat_train.reshape(-1, 1), y_train)
    pred = model.predict(lat_test.reshape(-1, 1))
    metrics = calculate_metrics(y_test, pred)
    metrics['model'] = 'Latitude only'
    results.append(metrics)

    # 2. Latitude + Longitude
    model = LinearRegression()
    model.fit(coords_train, y_train)
    pred = model.predict(coords_test)
    metrics = calculate_metrics(y_test, pred)
    metrics['model'] = 'Lat + Lon'
    results.append(metrics)

    # 3. MOSAIKS only
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    mosaiks_alpha = model.alpha_
    metrics = calculate_metrics(y_test, pred)
    metrics['model'] = 'MOSAIKS only'
    results.append(metrics)

    # 4. MOSAIKS + Latitude
    X_lat_train = np.column_stack([X_train_s, lat_train_s])
    X_lat_test = np.column_stack([X_test_s, lat_test_s])
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_lat_train, y_train)
    pred = model.predict(X_lat_test)
    metrics = calculate_metrics(y_test, pred)
    metrics['model'] = 'MOSAIKS + Lat'
    results.append(metrics)

    # 5. MOSAIKS + Latitude + Longitude
    X_geo_train = np.column_stack([X_train_s, lat_train_s, lon_train_s])
    X_geo_test = np.column_stack([X_test_s, lat_test_s, lon_test_s])
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_geo_train, y_train)
    pred = model.predict(X_geo_test)
    metrics = calculate_metrics(y_test, pred)
    metrics['model'] = 'MOSAIKS + Lat + Lon'
    results.append(metrics)

    return results, mosaiks_alpha


def run_spatial_cv(X, y, lat, lon, alphas):
    """Run spatial cross-validation experiments."""
    spatial_results = []

    # Standardize features once
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # 1. East/West Holdout (split at longitude -100)
    print("\n  East/West holdout...")
    west_mask = lon < -100
    east_mask = ~west_mask

    # Train on East, test on West
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_s[east_mask], y[east_mask])
    pred = model.predict(X_s[west_mask])
    metrics_ew = calculate_metrics(y[west_mask], pred)

    # Train on West, test on East
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_s[west_mask], y[west_mask])
    pred = model.predict(X_s[east_mask])
    metrics_we = calculate_metrics(y[east_mask], pred)

    # Average metrics
    avg_r2 = (metrics_ew['r2'] + metrics_we['r2']) / 2
    avg_rmse = (metrics_ew['rmse'] + metrics_we['rmse']) / 2
    avg_mae = (metrics_ew['mae'] + metrics_we['mae']) / 2
    spatial_results.append({
        'strategy': 'East/West holdout',
        'r2': avg_r2,
        'rmse': avg_rmse,
        'mae': avg_mae
    })

    # 2. Spatial Block CV (k-means clustering)
    print("  Spatial block CV (10 blocks)...")
    coords = np.column_stack([lat, lon])
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    blocks = kmeans.fit_predict(coords)

    block_r2s, block_rmses, block_maes = [], [], []
    for block_id in range(10):
        test_mask = blocks == block_id
        train_mask = ~test_mask

        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        # Re-fit scaler on training data only
        block_scaler = StandardScaler()
        X_train_s = block_scaler.fit_transform(X[train_mask])
        X_test_s = block_scaler.transform(X[test_mask])

        model = RidgeCV(alphas=alphas, cv=min(5, train_mask.sum()))
        model.fit(X_train_s, y[train_mask])
        pred = model.predict(X_test_s)
        metrics = calculate_metrics(y[test_mask], pred)
        block_r2s.append(metrics['r2'])
        block_rmses.append(metrics['rmse'])
        block_maes.append(metrics['mae'])

    spatial_results.append({
        'strategy': 'Spatial block CV',
        'r2': np.mean(block_r2s),
        'rmse': np.mean(block_rmses),
        'mae': np.mean(block_maes)
    })

    return spatial_results


def main(test_size=0.2, random_state=42, data_path=None, output_dir=None):
    print("=" * 60)
    print("MOSAIKS Temperature Prediction - Model Comparison")
    print("=" * 60)

    # Use provided paths or defaults
    input_file = Path(data_path) if data_path else INPUT_FILE
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR

    # Create output directory if needed
    out_dir.mkdir(exist_ok=True)

    # Load data
    data = pd.read_csv(input_file)
    print(f"\nDataset: {len(data)} grid cells")

    # Prepare features
    feature_cols = [f'X_{i}' for i in range(4000)]
    X = data[feature_cols].values
    y = data['avg_temp_c'].values
    lat = data['lat'].values
    lon = data['lon'].values

    # Train/test split using indices
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)

    print(f"Train set: {len(train_idx)} samples")
    print(f"Test set: {len(test_idx)} samples")

    alphas = np.logspace(-4, 6, 50)

    # Run model comparison
    print("\nTraining models...")
    results, mosaiks_alpha = run_model_comparison(X, y, lat, lon, train_idx, test_idx, alphas)

    # Print results
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS (Random Split)")
    print("=" * 60)
    print()

    print_metrics_header()
    for r in results:
        print_metrics_row(r['model'], r['r2'], r['rmse'], r['mae'])

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

    # Spatial Cross-Validation
    print("\n" + "=" * 60)
    print("SPATIAL GENERALIZATION")
    print("=" * 60)
    print("\nRunning spatial CV experiments...")

    spatial_results = run_spatial_cv(X, y, lat, lon, alphas)

    print("\n" + "-" * 60)
    print(f"{'Strategy':<25} {'R2':>10} {'RMSE (C)':>10} {'RMSE (F)':>10}")
    print("-" * 60)
    print(f"{'Random split (reference)':<25} {mosaiks_r2:>10.4f} {results[2]['rmse']:>10.2f} {results[2]['rmse']*9/5:>10.2f}")
    for sr in spatial_results:
        print(f"{sr['strategy']:<25} {sr['r2']:>10.4f} {sr['rmse']:>10.2f} {sr['rmse']*9/5:>10.2f}")

    r2_drop = mosaiks_r2 - spatial_results[1]['r2']
    print(f"\nR² drop under spatial CV: {r2_drop:.4f} ({r2_drop/mosaiks_r2*100:.1f}% relative)")
    print("This reflects the challenge of extrapolating to entirely unseen regions.")

    # Save results
    results_df = pd.DataFrame(results)
    output_file = out_dir / "model_comparison.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved model comparison to {output_file}")

    # Save spatial results
    spatial_df = pd.DataFrame(spatial_results)
    spatial_file = out_dir / "spatial_cv_results.csv"
    spatial_df.to_csv(spatial_file, index=False)
    print(f"Saved spatial CV results to {spatial_file}")

    return results, spatial_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate MOSAIKS temperature models with multiple configurations'
    )
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data for testing (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to input CSV')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        test_size=args.test_size,
        random_state=args.seed,
        data_path=args.data_path,
        output_dir=args.output_dir
    )
