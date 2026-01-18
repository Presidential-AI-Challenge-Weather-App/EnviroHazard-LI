import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N_SAMPLES = 200

TOWN_NAMES = [
    "Riverhead", "Southampton", "East Hampton", "Huntington", "Brookhaven",
    "Islip", "Smithtown", "Babylon", "Hempstead", "North Hempstead",
    "Oyster Bay", "Long Beach", "Glen Cove", "Freeport", "Port Jefferson",
    "Patchogue", "Bay Shore", "Lindenhurst", "West Islip", "Ronkonkoma"
]

SOIL_TYPES = ["Sandy", "Clay", "Loamy", "Silty", "Peaty", "Chalky", 
              "Sandy-Loam", "Clay-Loam", "Silty-Clay", "Rocky"]

WIND_DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def generate_base_features(n_samples):
    
    town_names = [TOWN_NAMES[i % len(TOWN_NAMES)] for i in range(n_samples)]
    
    latitude = np.random.uniform(40.6, 41.2, n_samples)
    longitude = np.random.uniform(-73.5, -71.8, n_samples)
    
    pop_density_type = np.random.choice([100.0, 1000.0, 5000.0], n_samples, p=[0.2, 0.5, 0.3])
    population_density = pop_density_type * np.random.uniform(0.5, 2.0, n_samples)
    
    urbanization_index = np.clip(population_density / 6000.0, 0.0, 1.0)
    urbanization_index = urbanization_index + np.random.normal(0, 0.1, n_samples)
    urbanization_index = np.clip(urbanization_index, 0.0, 1.0)
    
    elevation = np.random.gamma(2, 30, n_samples)
    elevation = np.clip(elevation, 0.0, 2500.0)
    slope = np.random.exponential(5, n_samples)
    slope = np.clip(slope, 0.0, 45.0)
    
    aspect = np.random.uniform(0, 360, n_samples)
    
    distance_to_water = np.random.exponential(5, n_samples)
    distance_to_water = np.clip(distance_to_water, 0.0, 50.0)
    
    impervious_surface_ratio = urbanization_index * np.random.uniform(0.5, 1.2, n_samples)
    impervious_surface_ratio = np.clip(impervious_surface_ratio, 0.0, 1.0)
    
    vegetation_density = (1.0 - urbanization_index) * np.random.uniform(0.5, 1.2, n_samples)
    vegetation_density = np.clip(vegetation_density, 0.0, 1.0)
    
    soil_type = np.random.choice(SOIL_TYPES, n_samples)
    
    return {
        'town_name': town_names,
        'latitude': latitude,
        'longitude': longitude,
        'population_density': population_density,
        'urbanization_index': urbanization_index,
        'elevation': elevation,
        'slope': slope,
        'aspect': aspect,
        'distance_to_water': distance_to_water,
        'impervious_surface_ratio': impervious_surface_ratio,
        'vegetation_density': vegetation_density,
        'soil_type': soil_type
    }


def generate_weather_features(n_samples):
    base_temp = np.random.choice([50.0, 70.0, 90.0], n_samples, p=[0.2, 0.5, 0.3])
    temperature = base_temp + np.random.normal(0, 10, n_samples)
    temperature = np.clip(temperature, 40.0, 110.0)
    
    humidity = 100.0 - (temperature - 40.0) * 0.6 + np.random.normal(0, 15, n_samples)
    humidity = np.clip(humidity, 10.0, 100.0)
    
    precip_24h_base = np.random.choice([0.0, 0.0, 0.0, 5.0, 20.0, 50.0], n_samples, p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    precip_24h = precip_24h_base + np.random.exponential(2, n_samples)
    precip_24h = np.clip(precip_24h, 0.0, 100.0)
    
    precip_7d = precip_24h + np.random.exponential(10, n_samples)
    precip_7d = np.clip(precip_7d, precip_24h, 300.0)
    
    wind_speed = np.random.gamma(3, 3, n_samples)
    wind_speed = np.clip(wind_speed, 0.0, 60.0)
    
    wind_direction = np.random.choice(WIND_DIRECTIONS, n_samples)
    
    days_since_rain = np.where(
        precip_24h > 1.0,
        0.0,
        np.random.exponential(7, n_samples)
    )
    days_since_rain = np.clip(days_since_rain, 0.0, 60.0).astype(int)
    
    avg_temp_past_week = temperature + np.random.normal(0, 5, n_samples)
    avg_temp_past_week = np.clip(avg_temp_past_week, 40.0, 110.0)
    
    max_temp_past_week = avg_temp_past_week + np.random.uniform(5, 15, n_samples)
    max_temp_past_week = np.clip(max_temp_past_week, temperature, 115.0)
    
    storm_probability = (precip_24h / 100.0) * 0.5 + (wind_speed / 60.0) * 0.5
    storm_warning_flag = (np.random.random(n_samples) < storm_probability).astype(int)
    
    recent_fire_flag = (np.random.random(n_samples) < 0.05).astype(int)
    
    return {
        'temperature': temperature,
        'humidity': humidity,
        'precipitation_last_24h': precip_24h,
        'precipitation_last_7d': precip_7d,
        'wind_speed': wind_speed,
        'wind_direction': wind_direction,
        'days_since_last_rain': days_since_rain,
        'avg_temp_past_week': avg_temp_past_week,
        'max_temp_past_week': max_temp_past_week,
        'storm_warning_flag': storm_warning_flag,
        'recent_fire_flag': recent_fire_flag
    }


def generate_usgs_noaa_features(df):
    precip_effect = np.clip(df['precipitation_last_7d'].values / 100.0, 0.0, 1.0) * 0.5
    temp_effect = (1.0 - (df['temperature'].values - 40.0) / 70.0) * 0.3
    dry_effect = (1.0 - np.clip(df['days_since_last_rain'].values / 30.0, 0.0, 1.0)) * 0.2
    soil_moisture = precip_effect + temp_effect + dry_effect
    soil_moisture = soil_moisture + np.random.normal(0, 0.05, len(df))
    soil_moisture = np.clip(soil_moisture, 0.0, 1.0)
    
    runoff_base = df['precipitation_last_24h'].values * 0.05  # Base runoff rate
    impervious_factor = 1.0 + df['impervious_surface_ratio'].values * 2.0  # More impervious = more runoff
    slope_factor = 1.0 + np.clip(df['slope'].values / 45.0, 0.0, 1.0) * 0.5  # Steeper slope = more runoff
    surface_runoff = runoff_base * impervious_factor * slope_factor
    surface_runoff = surface_runoff + np.random.exponential(0.5, len(df))
    surface_runoff = np.clip(surface_runoff, 0.0, 50.0)
    
    elevation_effect = 1.0 - np.clip(df['elevation'].values / 200.0, 0.0, 1.0)
    water_proximity_effect = 1.0 - np.clip(df['distance_to_water'].values / 20.0, 0.0, 1.0)
    precip_flow_effect = np.clip(df['precipitation_last_7d'].values / 150.0, 0.0, 1.0)
    streamflow_index = (elevation_effect * 0.4 + 
                       water_proximity_effect * 0.35 + 
                       precip_flow_effect * 0.25)
    streamflow_index = streamflow_index + np.random.normal(0, 0.08, len(df))
    streamflow_index = np.clip(streamflow_index, 0.0, 1.0)
    
    temp_drought = np.clip((df['temperature'].values - 60.0) / 50.0, 0.0, 1.0) * 0.35
    precip_drought = (1.0 - np.clip(df['precipitation_last_7d'].values / 100.0, 0.0, 1.0)) * 0.40
    dry_days_drought = np.clip(df['days_since_last_rain'].values / 30.0, 0.0, 1.0) * 0.25
    drought_index = temp_drought + precip_drought + dry_days_drought
    drought_index = drought_index + np.random.normal(0, 0.05, len(df))
    drought_index = np.clip(drought_index, 0.0, 1.0)
    
    return {
        'soil_moisture': soil_moisture,
        'surface_runoff': surface_runoff,
        'streamflow_index': streamflow_index,
        'drought_index': drought_index
    }


def calculate_risk_scores(df):
    fire_risk = np.zeros(len(df), dtype=np.float64)
    temp_norm = (df['temperature'].values - 40.0) / 70.0
    fire_risk += temp_norm * 0.20
    humidity_norm = (100.0 - df['humidity'].values) / 90.0
    fire_risk += humidity_norm * 0.18
    dry_days_norm = np.clip(df['days_since_last_rain'].values / 30.0, 0.0, 1.0)
    fire_risk += dry_days_norm * 0.15
    wind_norm = np.clip(df['wind_speed'].values / 40.0, 0.0, 1.0)
    fire_risk += wind_norm * 0.12
    fire_risk += df['vegetation_density'].values * 0.10
    fire_risk += df['drought_index'].values * 0.15
    fire_risk += (1.0 - df['soil_moisture'].values) * 0.08
    fire_risk += df['recent_fire_flag'].values.astype(np.float64) * 0.02
    fire_risk = np.clip(fire_risk, 0.0, 1.0)
    flood_risk = np.zeros(len(df), dtype=np.float64)
    precip_24h_norm = np.clip(df['precipitation_last_24h'].values / 80.0, 0.0, 1.0)
    flood_risk += precip_24h_norm * 0.22
    precip_7d_norm = np.clip(df['precipitation_last_7d'].values / 200.0, 0.0, 1.0)
    flood_risk += precip_7d_norm * 0.18
    elevation_norm = 1.0 - np.clip(df['elevation'].values / 100.0, 0.0, 1.0)
    flood_risk += elevation_norm * 0.15
    water_norm = 1.0 - np.clip(df['distance_to_water'].values / 20.0, 0.0, 1.0)
    flood_risk += water_norm * 0.10
    flood_risk += df['impervious_surface_ratio'].values * 0.08
    runoff_norm = np.clip(df['surface_runoff'].values / 30.0, 0.0, 1.0)
    flood_risk += runoff_norm * 0.12
    flood_risk += df['streamflow_index'].values * 0.10
    flood_risk += df['storm_warning_flag'].values.astype(np.float64) * 0.05
    flood_risk = np.clip(flood_risk, 0.0, 1.0)
    return fire_risk, flood_risk


def calculate_hazard_class(fire_risk, flood_risk):
    combined_risk = np.maximum(fire_risk, flood_risk)
    hazard_class = []
    for risk in combined_risk:
        if risk < 0.35:
            hazard_class.append("low")
        elif risk < 0.65:
            hazard_class.append("moderate")
        else:
            hazard_class.append("high")
    
    return hazard_class


def generate_dataset(n_samples=200):
    print("Generating realistic hazard dataset with USGS/NOAA-style data...")
    print(f"Number of samples: {n_samples}")
    
    base_features = generate_base_features(n_samples)
    weather_features = generate_weather_features(n_samples)
    
    df = pd.DataFrame({**base_features, **weather_features})
    
    usgs_noaa_features = generate_usgs_noaa_features(df)
    for key, value in usgs_noaa_features.items():
        df[key] = value
    
    fire_risk, flood_risk = calculate_risk_scores(df)
    df['fire_risk_score'] = fire_risk
    df['flood_risk_score'] = flood_risk
    
    df['hazard_class'] = calculate_hazard_class(fire_risk, flood_risk)
    
    column_order = [
        'town_name', 'latitude', 'longitude', 'population_density', 'urbanization_index',
        'temperature', 'humidity', 'precipitation_last_24h', 'precipitation_last_7d',
        'wind_speed', 'wind_direction', 'days_since_last_rain', 'avg_temp_past_week',
        'max_temp_past_week', 'storm_warning_flag',
        'elevation', 'slope', 'aspect', 'distance_to_water', 'impervious_surface_ratio',
        'vegetation_density', 'soil_type', 'soil_moisture', 'surface_runoff', 
        'streamflow_index', 'drought_index',
        'recent_fire_flag', 'fire_risk_score', 'flood_risk_score', 'hazard_class'
    ]
    
    df = df[column_order]
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].round(4)
    
    return df


def main():
    df = generate_dataset(n_samples=N_SAMPLES)
    
    output_file = "hazard_dummy_data_v2.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Dataset saved to: {output_file}")
    
    print(f"\nDataset Statistics:")
    print(f"  Total rows: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")
    print(f"\nHazard Class Distribution:")
    print(df['hazard_class'].value_counts())
    print(f"\nFire Risk Score:")
    print(f"  Min: {df['fire_risk_score'].min():.3f}")
    print(f"  Max: {df['fire_risk_score'].max():.3f}")
    print(f"  Mean: {df['fire_risk_score'].mean():.3f}")
    print(f"\nFlood Risk Score:")
    print(f"  Min: {df['flood_risk_score'].min():.3f}")
    print(f"  Max: {df['flood_risk_score'].max():.3f}")
    print(f"  Mean: {df['flood_risk_score'].mean():.3f}")
    print(f"\nUSGS/NOAA Environmental Variables:")
    print(f"  Soil Moisture - Mean: {df['soil_moisture'].mean():.3f}")
    print(f"  Surface Runoff - Mean: {df['surface_runoff'].mean():.3f} mm/hr")
    print(f"  Streamflow Index - Mean: {df['streamflow_index'].mean():.3f}")
    print(f"  Drought Index - Mean: {df['drought_index'].mean():.3f}")
    print("First 5 rows of generated data:")
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    print(df.head())


if __name__ == "__main__":
    main()