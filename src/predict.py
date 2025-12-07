import os
import sys
import requests
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Constants
SRC_DIR = Path(__file__).parent
GRID_FILE = SRC_DIR / "us_grid_025deg.csv"
MODEL_FILE = SRC_DIR / "output" / "model.joblib"

def get_lat_lon(address, api_key):
    """Geocode address using Google Maps API."""
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": api_key
    }
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

    data = response.json()
    if data['status'] != 'OK':
        raise Exception(f"Geocoding Error: {data['status']} - {data.get('error_message', '')}")

    if not data['results']:
        raise Exception("No results found for this address.")

    location = data['results'][0]['geometry']['location']
    return location['lat'], location['lng']

def find_nearest_grid_cell(lat, lon, grid_df):
    """Find the nearest grid cell in the dataframe."""
    # Grid centers are at X.5
    # Formula: round(coord - 0.5) + 0.5
    target_lat = np.round(lat - 0.5) + 0.5
    target_lon = np.round(lon - 0.5) + 0.5

    # Find matching row
    match = grid_df[(np.isclose(grid_df['lat'], target_lat)) &
                    (np.isclose(grid_df['lon'], target_lon))]

    if len(match) == 0:
        return None, target_lat, target_lon

    return match.iloc[0], target_lat, target_lon

def main():
    print("=" * 60)
    print("MOSAIKS Temperature Predictor")
    print("=" * 60)

    # Get API Key
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        print("Note: You can set GOOGLE_MAPS_API_KEY environment variable to avoid typing it.")
        api_key = input("Enter your Google Maps API Key: ").strip()
        if not api_key:
            print("Error: API Key is required.")
            return

    # Load Model
    if not MODEL_FILE.exists():
        print(f"Error: Model file not found at {MODEL_FILE}")
        print("Please run train.py first.")
        return

    print("Loading model...")
    try:
        model_data = joblib.load(MODEL_FILE)
        if isinstance(model_data, dict) and 'model' in model_data and 'scaler' in model_data:
            model = model_data['model']
            scaler = model_data['scaler']
        else:
            # Fallback if model file format is different (e.g. just the model)
            # But based on train.py, it is a dict.
            print("Error: Model file format is incorrect. Expected dictionary with 'model' and 'scaler'.")
            return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load Grid Data
    if not GRID_FILE.exists():
        print(f"Error: Grid file not found at {GRID_FILE}")
        return

    print("Loading grid data...")
    try:
        grid_df = pd.read_csv(GRID_FILE)
    except Exception as e:
        print(f"Error loading grid data: {e}")
        return

    while True:
        print("\n" + "-" * 40)
        address = input("Enter a street address (or 'q' to quit): ").strip()
        if address.lower() == 'q':
            break

        try:
            print(f"Geocoding '{address}'...")
            lat, lon = get_lat_lon(address, api_key)
            print(f"Coordinates: {lat:.4f}, {lon:.4f}")

            print("Finding nearest grid cell...")
            cell, grid_lat, grid_lon = find_nearest_grid_cell(lat, lon, grid_df)

            if cell is None:
                print(f"Error: No data found for grid cell centered at ({grid_lat}, {grid_lon}).")
                print("The location might be outside the US or the training data coverage.")
                continue

            print(f"Using grid cell centered at: {grid_lat}, {grid_lon}")

            # Extract features
            feature_cols = [f'X_{i}' for i in range(4000)]
            features = cell[feature_cols].values.reshape(1, -1)

            # Scale features
            features_scaled = scaler.transform(features)

            # Predict (model outputs Celsius)
            pred_temp_c = model.predict(features_scaled)[0]
            pred_temp_f = pred_temp_c * 9/5 + 32

            print(f"\nPredicted Average Temperature: {pred_temp_f:.1f}°F ({pred_temp_c:.1f}°C)")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
