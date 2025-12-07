"""
Prepare MOSAIKS 1x1 degree grid data with weather station temperature ground truth.

This script:
1. Loads the coarsened 1-degree MOSAIKS features
2. Filters to continental US
3. Matches with weather station temperature data
4. Outputs a clean dataset ready for regression
"""

import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print("MOSAIKS Temperature Data Preparation")
    print("=" * 60)

    # Load coarsened 1x1 degree MOSAIKS grid
    print("\n1. Loading MOSAIKS 1-degree grid...")
    grid = pd.read_csv("mosaiks_1deg_global.csv")
    print(f"   Global grid cells: {len(grid):,}")

    # Filter to continental US
    print("\n2. Filtering to continental US...")
    us_grid = grid[
        (grid['lat'] >= 24) & (grid['lat'] <= 50) &
        (grid['lon'] >= -125) & (grid['lon'] <= -66)
    ].copy()
    print(f"   US grid cells: {len(us_grid)}")

    # Load weather stations
    print("\n3. Loading weather stations...")
    weather = pd.read_csv("weather_stations.csv")
    print(f"   Total stations: {len(weather)}")

    # Filter by days recorded
    weather = weather[weather['days_recorded'] > 100].copy()
    print(f"   Stations with >100 days: {len(weather)}")

    # Round weather station coords to match 1-degree grid
    print("\n4. Aggregating temperature by grid cell...")
    weather['grid_lat'] = np.round(weather['latitude'])
    weather['grid_lon'] = np.round(weather['longitude'])

    # Average temperature per grid cell
    grid_temps = weather.groupby(['grid_lat', 'grid_lon']).agg({
        'yearly_avg_temp_c': 'mean',
        'station_id': 'count'
    }).reset_index()
    grid_temps.columns = ['lat', 'lon', 'avg_temp_c', 'num_stations']

    # Shift to match grid center (coordinates are at x.5)
    grid_temps['lat'] = grid_temps['lat'] + 0.5
    grid_temps['lon'] = grid_temps['lon'] + 0.5
    print(f"   Grid cells with temperature: {len(grid_temps)}")

    # Merge with MOSAIKS features
    print("\n5. Merging features with temperature...")
    data = us_grid.merge(grid_temps, on=['lat', 'lon'], how='inner')
    print(f"   Final dataset: {len(data)} grid cells")

    # Save
    output_file = "us_grid_1deg.csv"
    data.to_csv(output_file, index=False)
    print(f"\n   Saved to {output_file}")
    print(f"   Shape: {data.shape}")
    print(f"   Temperature range: {data['avg_temp_c'].min():.1f}C to {data['avg_temp_c'].max():.1f}C")

    return data


if __name__ == "__main__":
    main()
