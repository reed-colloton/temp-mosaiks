# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MOSAIKS temperature prediction project: predicts average yearly temperature from satellite imagery features using ridge regression. Achieves R² = 0.85 with 4,000 random convolutional features extracted from Planet satellite imagery (2019).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (outputs to src/output/)
python src/train.py

# Compare model configurations (latitude baseline vs MOSAIKS)
python src/evaluate.py

# Interactive prediction from address (requires GOOGLE_MAPS_API_KEY env var)
python src/predict.py
```

## Architecture

**Data flow:**
- Input data: `data/us_grid_025deg.csv` (1,709 US grid cells with 4,000 MOSAIKS features + temp labels)
- Global features for predictions: `data/global_grid_1deg.csv`
- Trained model and predictions saved to `src/output/`

**Scripts:**
- `src/train.py` - Trains RidgeCV model on MOSAIKS features, saves model.joblib and test_predictions.csv
- `src/evaluate.py` - Compares 5 model variants (lat-only, lat+lon, MOSAIKS, MOSAIKS+lat, MOSAIKS+lat+lon)
- `src/predict.py` - Interactive CLI for predicting temperature from street addresses using Google Maps geocoding

**Model:** StandardScaler + RidgeCV with cross-validated alpha selection (alphas from 10^-4 to 10^6).

## Data Format

Features are columns `X_0` through `X_3999` (normalized values 0-0.5). Target is `avg_temp_c`. Each row represents a 0.25° × 0.25° grid cell (~25 km resolution).

## GitHub Pages

The repo includes Jekyll configuration (`_config.yml`, `_layouts/`) for GitHub Pages hosting with the minimal theme. README.md serves as the landing page.
