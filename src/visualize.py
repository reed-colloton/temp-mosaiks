"""
Generate visualizations for MOSAIKS temperature predictions.

Creates:
- Scatter plot of actual vs predicted temperatures
- US map showing prediction errors

Usage:
    python visualize.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
SRC_DIR = Path(__file__).parent
OUTPUT_DIR = SRC_DIR / "output"
PREDICTIONS_FILE = OUTPUT_DIR / "test_predictions.csv"


def create_scatter_plot(df, output_path):
    """Create scatter plot of actual vs predicted temperatures."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Calculate error for coloring
    errors = np.abs(df['error_c'])

    scatter = ax.scatter(
        df['actual_temp_c'],
        df['predicted_temp_c'],
        c=errors,
        cmap='RdYlGn_r',
        alpha=0.7,
        s=50,
        edgecolors='white',
        linewidth=0.5
    )

    # Add 1:1 reference line
    min_temp = min(df['actual_temp_c'].min(), df['predicted_temp_c'].min())
    max_temp = max(df['actual_temp_c'].max(), df['predicted_temp_c'].max())
    ax.plot([min_temp, max_temp], [min_temp, max_temp], 'k--', alpha=0.5, label='Perfect prediction')

    # Labels and title
    ax.set_xlabel('Actual Temperature (°C)', fontsize=12)
    ax.set_ylabel('Predicted Temperature (°C)', fontsize=12)
    ax.set_title('MOSAIKS Temperature Prediction: Actual vs Predicted', fontsize=14)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Absolute Error (°C)', fontsize=10)

    # Add R² annotation
    r2 = 1 - (np.sum((df['actual_temp_c'] - df['predicted_temp_c'])**2) /
              np.sum((df['actual_temp_c'] - df['actual_temp_c'].mean())**2))
    rmse = np.sqrt(np.mean(df['error_c']**2))
    ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.2f}°C',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend(loc='lower right')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plot to {output_path}")


def create_error_map(df, output_path):
    """Create US map showing prediction errors."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot error with diverging colormap (blue=underpredict, red=overpredict)
    scatter = ax.scatter(
        df['lon'],
        df['lat'],
        c=df['error_c'],
        cmap='RdBu',
        vmin=-5,
        vmax=5,
        s=80,
        alpha=0.8,
        edgecolors='black',
        linewidth=0.3
    )

    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Temperature Prediction Errors Across US Test Set\n(Red = overpredicted, Blue = underpredicted)', fontsize=14)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Prediction Error (°C)', fontsize=10)

    # Set reasonable US bounds
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved error map to {output_path}")


def create_predicted_temp_map(df, output_path):
    """Create US map showing predicted temperatures."""
    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(
        df['lon'],
        df['lat'],
        c=df['predicted_temp_c'],
        cmap='RdYlBu_r',
        s=80,
        alpha=0.8,
        edgecolors='black',
        linewidth=0.3
    )

    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Predicted Annual Average Temperature (Test Set)', fontsize=14)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Temperature (°C)', fontsize=10)

    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved temperature map to {output_path}")


def main():
    print("=" * 60)
    print("MOSAIKS Temperature Prediction - Visualization")
    print("=" * 60)

    # Check for predictions file
    if not PREDICTIONS_FILE.exists():
        print(f"Error: Predictions file not found at {PREDICTIONS_FILE}")
        print("Please run train.py first to generate predictions.")
        return

    # Load predictions
    print(f"\nLoading predictions from {PREDICTIONS_FILE}")
    df = pd.read_csv(PREDICTIONS_FILE)
    print(f"Loaded {len(df)} test predictions")

    # Create output directory if needed
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Generate visualizations
    print("\nGenerating visualizations...")

    scatter_path = OUTPUT_DIR / "scatter_actual_vs_predicted.png"
    create_scatter_plot(df, scatter_path)

    error_map_path = OUTPUT_DIR / "map_prediction_errors.png"
    create_error_map(df, error_map_path)

    temp_map_path = OUTPUT_DIR / "map_predicted_temps.png"
    create_predicted_temp_map(df, temp_map_path)

    print("\nDone! Generated 3 visualizations.")


if __name__ == "__main__":
    main()
