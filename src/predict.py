import os
import re
import requests
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Constants
SRC_DIR = Path(__file__).parent
DATA_DIR = SRC_DIR.parent / "data"
GRID_FILE = DATA_DIR / "global_grid_1deg.csv"  # Global 1° grid for predictions
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
    formatted_address = data['results'][0]['formatted_address']
    return location['lat'], location['lng'], formatted_address

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in km using haversine formula."""
    R = 6371  # Earth radius in km

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def find_nearest_grid_cell(lat, lon, grid_df):
    """Find the nearest available grid cell in the dataframe using haversine distance."""
    distances_km = haversine_distance(lat, lon, grid_df['lat'].values, grid_df['lon'].values)
    nearest_idx = np.argmin(distances_km)
    nearest_cell = grid_df.iloc[nearest_idx]
    distance_km = distances_km[nearest_idx]

    return nearest_cell, nearest_cell['lat'], nearest_cell['lon'], distance_km


def parse_coordinates(input_str):
    """
    Try to parse input as coordinates (lat, lon).
    Returns (lat, lon) tuple if valid, None otherwise.
    """
    # Pattern: optional minus, digits, optional decimal, comma, same pattern
    coord_pattern = r'^(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)$'
    match = re.match(coord_pattern, input_str.strip())
    if match:
        lat = float(match.group(1))
        lon = float(match.group(2))
        # Basic validation
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    return None

def main():
    print("=" * 60)
    print("MOSAIKS Temperature Predictor")
    print("=" * 60)

    # Get API Key (optional - only needed for address geocoding)
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        print("\nNote: No GOOGLE_MAPS_API_KEY found.")
        print("You can enter coordinates directly (e.g., '39.7392, -104.9903')")
        print("or provide an API key to geocode street addresses.")

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

    try:
        grid_df = pd.read_csv(GRID_FILE)
    except Exception as e:
        print(f"Error loading grid data: {e}")
        return

    while True:
        print("\n" + "-" * 40)
        user_input = input("Enter address or coordinates (lat, lon), or 'q' to quit: ").strip()
        if user_input.lower() == 'q':
            break

        try:
            # Check if input is coordinates
            coords = parse_coordinates(user_input)
            if coords:
                lat, lon = coords
                formatted_address = f"Coordinates: {lat}, {lon}"
                print(f"\nLocation: {formatted_address}")
            else:
                # Try geocoding as address
                if not api_key:
                    api_key = input("Enter Google Maps API Key for address lookup: ").strip()
                    if not api_key:
                        print("Error: API key required for address lookup. Use coordinates instead.")
                        continue
                lat, lon, formatted_address = get_lat_lon(user_input, api_key)
                print(f"\nLocation: {formatted_address}")
                print(f"Coordinates: {lat:.4f}, {lon:.4f}")

            cell, grid_lat, grid_lon, distance_km = find_nearest_grid_cell(lat, lon, grid_df)
            print(f"Using grid cell at: ({grid_lat}, {grid_lon}) - {distance_km:.0f} km away")

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
