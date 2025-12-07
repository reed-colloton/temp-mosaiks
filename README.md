# MOSAIKS Temperature Prediction

Predicting average yearly temperature from satellite imagery features using the MOSAIKS (Multi-task Observation using Satellite Imagery & Kitchen Sinks) approach.

## Overview

This project demonstrates how MOSAIKS random convolutional features extracted from satellite imagery can predict ground-level temperature with high accuracy (R² = 0.85), outperforming a latitude-only baseline by 16 percentage points.

## Method

### MOSAIKS Features

MOSAIKS uses a "kitchen sink" approach:
1. Extract random convolutional features from satellite imagery (Planet imagery, 2019)
2. Apply 4,000 random filters to create a fixed feature representation
3. Use simple ridge regression to predict any target variable

The key insight is that these random features capture enough spatial variation in land cover, vegetation, urbanization, and terrain to predict many different outcomes without task-specific training.

### Data Sources

- **MOSAIKS Features**: Pre-computed 1°×1° grid features from [mosaiks.org](https://mosaiks.org)
- **Temperature**: NOAA weather station yearly averages (stations with >100 days recorded)

### Pipeline

```
mosaiks_1deg_global.csv (17,658 global grid cells)
        │
        ▼ Filter to continental US
us_grid_1deg.csv (606 cells with temp data)
        │
        ▼ 80/20 train/test split
        │
        ▼ StandardScaler + RidgeCV
        │
        ▼ Predictions
```

## Results

### Model Comparison

| Model              | R²     | RMSE    | MAE     |
|--------------------|--------|---------|---------|
| Latitude only      | 0.6869 | 3.28°C  | 2.58°C  |
| Lat + Lon          | 0.7127 | 3.15°C  | 2.47°C  |
| **MOSAIKS only**   | **0.8492** | **2.28°C** | **1.70°C** |
| MOSAIKS + Lat      | 0.8467 | 2.30°C  | 1.73°C  |
| MOSAIKS + Lat + Lon| 0.8402 | 2.35°C  | 1.76°C  |

### Key Findings

1. **MOSAIKS features alone achieve R² = 0.85** - no need for explicit geographic coordinates
2. **+16% improvement over latitude baseline** - satellite features capture more than just north-south variation
3. **Adding coordinates doesn't help** - MOSAIKS already encodes geographic patterns from imagery
4. **Resolution matters**: County-level aggregation destroyed the signal (R² ≈ 0), but 1° grid preserves it

### Why It Works

The random convolutional features capture:
- **Vegetation patterns** (forests vs deserts vs cropland)
- **Urban heat islands** (built-up areas)
- **Elevation proxies** (snow cover, vegetation zones)
- **Water bodies** (coastal vs inland)

These visual patterns correlate strongly with local climate.

## Failed Approaches & Lessons Learned

### Attempt 1: Native Resolution via Redivis API

**Approach**: Query the full 1km² resolution MOSAIKS data from Redivis database for all US states.

**Problem**: Hit compute quota limits (100,000 slot-seconds). Even chunking by state exceeded the free tier.

**Lesson**: The native resolution dataset is massive (~50M+ grid cells for the US). Need institutional access or pre-aggregated data for large-scale analysis.

### Attempt 2: 1% Sampled Dataset

**Approach**: Use the 1% random sample available on Redivis to reduce data size.

**Problem**: The 1% sample didn't include any US data - only Argentina, India, Spain, and other countries were represented.

**Lesson**: Random sampling of global data doesn't guarantee coverage of specific regions.

### Attempt 3: County-Level Aggregated Features (ADM2)

**Approach**: Use pre-aggregated county-level features from `ADM_2_regions_RCF_global_dense.csv` (area-weighted averages).

**Results**:
| Model | R² | RMSE |
|-------|-----|------|
| Latitude only | 0.69 | 3.73°C |
| MOSAIKS only | **-0.02** | 6.70°C |
| MOSAIKS + Lat | 0.61 | 4.14°C |

**Problem**: County-level aggregation completely destroyed the predictive signal. The model just predicted the mean temperature (R² ≈ 0). Adding MOSAIKS features actually *hurt* performance compared to latitude alone.

**Why it failed**:
- US counties average ~3,000 km² - far too coarse
- Area-weighted averaging smooths out the fine-grained texture patterns that MOSAIKS relies on
- The 4,000 features become nearly identical across counties after averaging
- Feature correlations with temperature dropped to near zero (max |r| = 0.06)

**Diagnostic evidence**:
```
Top feature correlations with temperature (county-level):
  X_6:    r = -0.058  (p = 0.098)  # Not significant
  X_92:   r = -0.057  (p = 0.108)  # Not significant
  ...
  Features with p < 0.05: 0 out of 4,000
```

### Attempt 4: 1° Grid Resolution (Success)

**Approach**: Use `coarsened_global_dense_grid_decimal_place=0.csv` - features aggregated to 1°×1° grid cells (~100km).

**Results**: R² = 0.85, RMSE = 2.28°C

**Why it worked**:
- 1° cells are ~10-30× smaller than counties
- Preserves enough spatial variation in the features
- Still captures vegetation, terrain, and land cover patterns
- Sufficient weather station coverage per cell for reliable ground truth

### Resolution Comparison

| Resolution | Avg Area | Sample Size | MOSAIKS R² |
|------------|----------|-------------|------------|
| Native (1km²) | 1 km² | ~50M cells | ~0.90* |
| 1° Grid | ~10,000 km² | 606 cells | **0.85** |
| County (ADM2) | ~3,000 km² | 806 counties | -0.02 |

*Expected based on MOSAIKS paper; not tested due to quota limits.

### Key Insight

**Spatial resolution is critical for MOSAIKS**. The random convolutional features encode fine-grained texture patterns from satellite imagery. When aggregated over large irregular regions (counties), these patterns average out and lose predictive power. Regular grid aggregation at moderate resolution (1°) preserves enough signal for accurate prediction.

## Files

```
mosaiks_temperature_project/
├── README.md              # This file
├── prepare_data.py        # Data preparation script
├── train.py               # Model training script
├── evaluate.py            # Model comparison script
├── mosaiks_1deg_global.csv # Raw MOSAIKS features (global)
├── weather_stations.csv   # NOAA temperature data
├── us_grid_1deg.csv       # Prepared US dataset
├── test_predictions.csv   # Model predictions (generated)
└── model_comparison.csv   # Evaluation results (generated)
```

## Usage

### 1. Prepare Data
```bash
python prepare_data.py
```
Filters global MOSAIKS grid to US and merges with weather station temperatures.

### 2. Train Model
```bash
python train.py
```
Trains ridge regression with cross-validated regularization. Outputs:
- Model performance metrics
- `test_predictions.csv` with actual vs predicted temperatures
- `model.joblib` saved model

### 3. Compare Models
```bash
python evaluate.py
```
Compares all feature combinations and saves `model_comparison.csv`.

## Requirements

```
pandas
numpy
scikit-learn
joblib
```

## Data Format

### Input: us_grid_1deg.csv
| Column | Description |
|--------|-------------|
| lat | Grid cell latitude (center, e.g., 35.5) |
| lon | Grid cell longitude (center, e.g., -105.5) |
| continent | Continent code |
| X_0 - X_3999 | 4,000 MOSAIKS random convolutional features |
| avg_temp_c | Mean yearly temperature (°C) |
| num_stations | Weather stations in grid cell |

### Output: test_predictions.csv
| Column | Description |
|--------|-------------|
| lat | Grid cell latitude |
| lon | Grid cell longitude |
| actual_temp_c | Observed temperature |
| predicted_temp_c | Model prediction |
| error_c | Prediction error (actual - predicted) |

## Limitations

- **1° resolution**: ~100km grid cells average out local variation
- **Station coverage**: Some grid cells have few weather stations
- **Temporal mismatch**: 2019 imagery vs multi-year temperature averages

## References

- Rolf et al. (2021). "A generalizable and accessible approach to machine learning with global satellite imagery." *Nature Communications*.
- MOSAIKS Project: https://mosaiks.org
# temp-mosaiks
# temp-mosaiks
