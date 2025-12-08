"""
Shared utility functions for MOSAIKS temperature prediction.
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def c_to_f(celsius):
    """Convert Celsius to Fahrenheit."""
    return celsius * 9/5 + 32


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics.

    Returns:
        dict with r2, rmse, mae (all in Celsius)
    """
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred)
    }


def print_metrics_header():
    """Print metrics table header."""
    print(f"{'Model':<25} {'R2':>10} {'RMSE (C)':>10} {'RMSE (F)':>10} {'MAE (C)':>10} {'MAE (F)':>10}")
    print("-" * 75)


def print_metrics_row(name, r2, rmse_c, mae_c):
    """Print formatted metrics row with C and F conversions."""
    rmse_f = rmse_c * 9/5
    mae_f = mae_c * 9/5
    print(f"{name:<25} {r2:>10.4f} {rmse_c:>10.2f} {rmse_f:>10.2f} {mae_c:>10.2f} {mae_f:>10.2f}")


def print_metrics(name, metrics):
    """Print metrics from a metrics dict."""
    print_metrics_row(name, metrics['r2'], metrics['rmse'], metrics['mae'])
