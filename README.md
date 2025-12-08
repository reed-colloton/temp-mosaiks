# MOSAIKS Temperature Prediction

Predicting average yearly temperature from satellite imagery features using the MOSAIKS (Multi-task Observation using Satellite Imagery & Kitchen Sinks) approach.

## Overview

This project demonstrates how MOSAIKS random convolutional features extracted from satellite imagery can predict ground-level temperature with high accuracy (R² = 0.85), outperforming a latitude-only baseline by 28.5 percentage points.

## Method

### MOSAIKS Features

MOSAIKS uses a "kitchen sink" approach:
1. Extract random convolutional features from satellite imagery (Planet imagery, 2019)
2. Apply 4,000 random filters to create a fixed feature representation
3. Use simple ridge regression to predict any target variable

The key insight is that these random features capture enough spatial variation in land cover, vegetation, urbanization, and terrain to predict many different outcomes without task-specific training.

### Data Sources

- **MOSAIKS Features**: Pre-computed 0.25°×0.25° grid features from [mosaiks.org](https://mosaiks.org)
- **Temperature**: NOAA weather station yearly averages (stations with >100 days recorded)

### Pipeline

```
us_grid_025deg.csv (1,709 US grid cells with features + temp)
        |
        v  80/20 train/test split
        |
        v  StandardScaler + RidgeCV
        |
        v  Predictions (in Celsius, displayed in F and C)
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **R²** (R-squared) | Coefficient of determination. Measures how well the model explains variance in temperature. R²=1.0 is perfect prediction; R²=0 means the model is no better than predicting the mean. |
| **RMSE** (Root Mean Square Error) | Square root of the average squared prediction error. Penalizes large errors more heavily. Reported in both °C and °F. |
| **MAE** (Mean Absolute Error) | Average absolute difference between predicted and actual temperature. More interpretable than RMSE - represents the typical error magnitude. Reported in both °C and °F. |

## Results

### Model Comparison (0.25° Grid, 1,709 cells)

| Model              | R²     | RMSE (C) | RMSE (F) | MAE (C) | MAE (F) |
|--------------------|--------|----------|----------|---------|---------|
| Latitude only      | 0.5609 | 3.92     | 7.06     | 3.14    | 5.66    |
| Lat + Lon          | 0.5755 | 3.85     | 6.94     | 3.02    | 5.44    |
| **MOSAIKS only**   | **0.8457** | **2.32** | **4.18** | **1.84** | **3.30** |
| MOSAIKS + Lat      | 0.8438 | 2.34     | 4.21     | 1.82    | 3.27    |
| MOSAIKS + Lat + Lon| 0.8477 | 2.31     | 4.16     | 1.81    | 3.25    |

### Key Findings

1. **MOSAIKS features alone achieve R² = 0.85** - no explicit coordinates needed
2. **+28.5% improvement over latitude baseline** - MOSAIKS captures local variation that coordinates miss
3. **Adding coordinates doesn't help** - MOSAIKS already encodes geographic patterns from imagery
4. **Resolution matters**: County-level aggregation destroyed the signal (R² ≈ 0), but grid-based aggregation works

### Why It Works

The random convolutional features capture:
- **Vegetation patterns** (forests vs deserts vs cropland)
- **Urban heat islands** (built-up areas)
- **Elevation proxies** (snow cover, vegetation zones)
- **Water bodies** (coastal vs inland)

These visual patterns correlate strongly with local climate.

## Files

```
temp-mosaiks/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore (excludes large data)
└── src/
    ├── train.py               # Model training script
    ├── evaluate.py            # Model comparison script
    ├── predict.py             # Predict temperature from address
    ├── us_grid_025deg.csv     # Training data (0.25° grid with temp labels)
    ├── global_grid_1deg.csv   # Full global MOSAIKS features (1° grid, for predictions)
    └── output/
        ├── model.joblib           # Trained model (generated)
        ├── test_predictions.csv   # Model predictions (generated)
        └── model_comparison.csv   # Evaluation results (generated)
```

## Usage

### 1. Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Train Model
Train the ridge regression model:
```bash
python src/train.py
```

### 3. Evaluate
Compare different feature sets:
```bash
python src/evaluate.py
```

### 4. Predict from Address
Predict temperature for any US street address using the trained model:
```bash
python src/predict.py
```
*Note: Requires a Google Maps API Key.*

## Requirements

```
pandas
numpy
scikit-learn
joblib
requests
python-dotenv
```

## Data Format

### Input: src/us_grid_025deg.csv
| Column | Description |
|--------|-------------|
| lat | Grid cell latitude (center, e.g., 35.625) |
| lon | Grid cell longitude (center, e.g., -105.375) |
| continent | Continent code |
| X_0 - X_3999 | 4,000 MOSAIKS random convolutional features |
| avg_temp_c | Mean yearly temperature (°C) |
| num_stations | Weather stations in grid cell |

### Output: src/output/test_predictions.csv
| Column | Description |
|--------|-------------|
| lat | Grid cell latitude |
| lon | Grid cell longitude |
| actual_temp_c | Observed temperature (°C) |
| predicted_temp_c | Model prediction (°C) |
| error_c | Prediction error in °C |
| actual_temp_f | Observed temperature (°F) |
| predicted_temp_f | Model prediction (°F) |
| error_f | Prediction error in °F |

## Limitations

- **0.25° resolution**: ~25km grid cells still average out some local variation
- **Station coverage**: Some grid cells have few weather stations
- **Temporal mismatch**: 2019 imagery vs multi-year temperature averages

## References

- Rolf et al. (2021). "A generalizable and accessible approach to machine learning with global satellite imagery." *Nature Communications*.
- MOSAIKS Project: https://mosaiks.org
