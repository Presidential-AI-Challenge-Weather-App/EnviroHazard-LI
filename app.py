#!/usr/bin/env python3
"""
EnviroHazardLI - Environmental hazard monitor for Long Island
A Flask web application that assesses fire and flood risks for selected towns
Run: python app.py
"""

import os
import re
import time
import json
import logging
import requests
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template, request

# If TensorFlow is installed, we'll use it; if not, the app falls back to rules
try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

app = Flask(__name__, template_folder="templates")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("envirohazardli")

# Paths for optional model files. If these files aren't present, we'll use rule-based fallbacks.
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "hazard_multioutput.h5"
ENCODERS_FILE = MODEL_DIR / "scaler_and_encoders.pkl"

# These are the runtime handles for model and preprocessing objects.
ml_model = None
std_scaler = None
onehot_encoder = None
label_encoder = None
feat_names = None
numeric_features = []
categorical_features = []

# Predefined towns with coordinates. 
TOWNS = {
   "Commack": (40.8429, -73.2920),
   "Smithtown": (40.8559, -73.2007),
   "Huntington": (40.8682, -73.4257),
   "Babylon": (40.7009, -73.3257),
   "Islip": (40.7301, -73.2104),
   "Southampton": (40.8843, -72.3895),
   "Riverhead": (40.9170, -72.6620),
   "Patchogue": (40.7651, -73.0151),
   "Bay Shore": (40.7251, -73.2451),
   "Centereach": (40.8565, -73.0818),
   "Hicksville": (40.7682, -73.5251),
   "Hempstead": (40.7062, -73.6187)
}

# Simple in-memory caches so we don't hammer external APIs during development.
noaa_cache = {}
elev_cache = {}
CACHE_SECONDS = 600  # 10 minutes

def init_model_and_encoders():
    
    global ml_model, std_scaler, onehot_encoder, label_encoder, feat_names, numeric_features, categorical_features

    ml_model = None
    std_scaler = None
    onehot_encoder = None
    label_encoder = None
    feat_names = None
    numeric_features = []
    categorical_features = []

    # Attempt to load the Keras model if present
    if MODEL_FILE.exists() and load_model is not None:
        try:
            logger.info("Loading ML model from %s", MODEL_FILE)
            ml_model = load_model(str(MODEL_FILE))
            logger.info("ML model loaded successfully")
        except Exception as exc:
            logger.warning("Could not load ML model: %s", exc)
            ml_model = None
    else:
        logger.info("No ML model file found or TensorFlow is unavailable")

    # Attempt to load scaler / encoders
    if ENCODERS_FILE.exists():
        try:
            with open(ENCODERS_FILE, "rb") as fh:
                enc = pickle.load(fh)
            std_scaler = enc.get("scaler")
            onehot_encoder = enc.get("ohe")
            label_encoder = enc.get("label_enc")
            feat_names = enc.get("feature_names")
            numeric_features = enc.get("num_cols", [])
            categorical_features = enc.get("cat_cols", [])

            # If feature names exist but the numeric/categorical lists are missing, try to infer them
            if feat_names and (not numeric_features or not categorical_features):
                if onehot_encoder is not None and hasattr(onehot_encoder, "feature_names_in_"):
                    categorical_features = list(onehot_encoder.feature_names_in_)
                    try:
                        ohe_out = list(onehot_encoder.get_feature_names_out())
                        numeric_features = [f for f in feat_names if f not in ohe_out]
                    except Exception:
                        numeric_features = feat_names
                else:
                    numeric_features = feat_names
                    categorical_features = []
            logger.info("Encoders loaded. numeric=%d categorical=%d", len(numeric_features), len(categorical_features))
        except Exception as exc:
            logger.warning("Failed to load encoders: %s", exc)
    else:
        logger.info("Encoders file not found at %s", ENCODERS_FILE)

# Initialize once at startup
init_model_and_encoders()

def get_noaa_point(lat, lon):
    """
    Fetch NOAA forecast metadata and forecast/hourly payloads for a lat/lon.
    Cache results briefly to avoid overloading the API.
    """
    cache_key = f"noaa_{lat:.4f}_{lon:.4f}"
    cached = noaa_cache.get(cache_key)
    if cached:
        data, ts = cached
        if time.time() - ts < CACHE_SECONDS:
            return data

    headers = {"User-Agent": "(EnviroHazardLI, contact@example.com)"}
    try:
        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        r = requests.get(points_url, headers=headers, timeout=10)
        if r.status_code == 404:
            logger.info("NOAA: points endpoint returned 404 for %s,%s", lat, lon)
            return None
        r.raise_for_status()
        props = r.json().get("properties", {})
        forecast_url = props.get("forecast")
        hourly_url = props.get("forecastHourly")
        forecast = None
        hourly = None
        if forecast_url:
            time.sleep(0.35)  # be gentle with the API
            r2 = requests.get(forecast_url, headers=headers, timeout=10)
            r2.raise_for_status()
            forecast = r2.json()
        if hourly_url:
            time.sleep(0.35)
            r3 = requests.get(hourly_url, headers=headers, timeout=10)
            r3.raise_for_status()
            hourly = r3.json()
        data = {"forecast": forecast, "hourly": hourly}
        noaa_cache[cache_key] = (data, time.time())
        return data
    except Exception as exc:
        logger.warning("NOAA fetch error: %s", exc)
        return None

def get_elevation(lat, lon):
    """
    Query USGS elevation API. Cache results locally.
    Returns elevation in meters; if the API fails, return a reasonable default.
    """
    cache_key = f"elev_{lat:.4f}_{lon:.4f}"
    if cache_key in elev_cache:
        return elev_cache[cache_key]
    try:
        url = "https://epqs.nationalmap.gov/v1/json"
        params = {"x": lon, "y": lat, "units": "Meters", "output": "json"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        val = data.get("value")
        if val is None:
            val = data.get("USGS_Elevation_Point_Query_Service", {}).get("Elevation_Query", {}).get("Elevation")
        if val is None:
            return 20.0
        elev_cache[cache_key] = float(val)
        return float(val)
    except Exception as exc:
        logger.warning("Elevation fetch error: %s", exc)
        return 20.0

def extract_weather_features(lat, lon, points_json):
    """
    Parse NOAA forecast/hourly JSON and derive a small set of weather features
    we use for downstream logic and (optionally) the ML model.
    Returns: (features_dict, hourly_periods_list)
    """
    default = {
        "temperature": 70.0,
        "humidity": 60.0,
        "wind_speed": 5.0,
        "precipitation_last_24h": 0.0,
        "precipitation_last_7d": 0.0,
        "days_since_last_rain": 3
    }

    if points_json is None:
        return default, None

    forecast = points_json.get("forecast")
    hourly = points_json.get("hourly")

    temperature = default["temperature"]
    humidity = default["humidity"]
    wind_speed = default["wind_speed"]
    precip24 = default["precipitation_last_24h"]
    precip7 = default["precipitation_last_7d"]
    days_since = default["days_since_last_rain"]

    try:
        periods = forecast.get("properties", {}).get("periods", []) if forecast else []
        if periods:
            current = periods[0]
            temperature = current.get("temperature", temperature)
            wind_raw = current.get("windSpeed", "")
            m = re.search(r"(\d+)", str(wind_raw))
            if m:
                wind_speed = float(m.group(1))
            dew = current.get("dewpoint", {}).get("value")
            if dew is not None:
                dew_f = dew
                if dew < 60:
                    dew_f = dew * 9.0/5.0 + 32.0
                humidity = max(10.0, min(100.0, 100 - 5*(temperature - dew_f)))
            short_text = current.get("shortForecast", "") if current else ""
            detailed_text = current.get("detailedForecast", "") if current else ""
        else:
            short_text = ""
            detailed_text = ""

        rainy_hours = 0
        hourly_periods = hourly.get("properties", {}).get("periods", [])[:24] if hourly else []
        rain_keywords = ["rain", "shower", "storm", "thunder", "drizzle"]
        for h in hourly_periods:
            txt = (h.get("shortForecast", "") + " " + h.get("detailedForecast", "")).lower()
            if any(k in txt for k in rain_keywords):
                rainy_hours += 1

        if "heavy" in (detailed_text.lower() if detailed_text else ""):
            precip24 = max(precip24, 25.0)
        elif "moderate" in (detailed_text.lower() if detailed_text else ""):
            precip24 = max(precip24, 10.0)
        elif rainy_hours > 0:
            precip24 = max(precip24, rainy_hours * 2.0)

        precip7 = max(precip7, precip24 * 3.0)
        days_since = 0 if precip24 > 0 else 3

        features = {
            "temperature": float(temperature),
            "humidity": float(humidity),
            "wind_speed": float(wind_speed),
            "precipitation_last_24h": float(precip24),
            "precipitation_last_7d": float(precip7),
            "days_since_last_rain": int(days_since),
            "shortForecast": short_text,
            "detailedForecast": detailed_text,
            "rainy_hours": rainy_hours
        }
        return features, hourly_periods
    except Exception as exc:
        logger.warning("Error extracting weather features: %s", exc)
        return default, []

def estimate_geo_features(lat, lon, elevation, weather_features):
    """
    Make some quick geographic/environmental estimates based on location and weather.
    These are heuristic and designed to mimic what a richer dataset would provide.
    """
    slope = float(np.random.uniform(0, 6))
    aspect = float(np.random.uniform(0, 360))
    center_lon = -73.0
    distance_to_water = max(0.5, min(15.0, abs(lon - center_lon) * 50.0))

    urban_centers = [(40.7062, -73.6187), (40.8682, -73.4257)]
    min_dist = min(np.sqrt((lat - ulat)**2 + (lon - ulon)**2) for ulat, ulon in urban_centers)

    if min_dist < 0.05:
        pop_density = float(np.random.uniform(3000, 8000))
        urban_index = float(np.random.uniform(0.7, 0.9))
        imperv = float(np.random.uniform(0.6, 0.9))
        veg = float(np.random.uniform(0.1, 0.3))
    elif min_dist < 0.15:
        pop_density = float(np.random.uniform(1000, 3000))
        urban_index = float(np.random.uniform(0.4, 0.7))
        imperv = float(np.random.uniform(0.3, 0.6))
        veg = float(np.random.uniform(0.3, 0.6))
    else:
        pop_density = float(np.random.uniform(100, 1000))
        urban_index = float(np.random.uniform(0.1, 0.4))
        imperv = float(np.random.uniform(0.1, 0.3))
        veg = float(np.random.uniform(0.6, 0.9))

    temp = weather_features.get("temperature", 70.0)
    humidity = weather_features.get("humidity", 60.0)
    precip = weather_features.get("precipitation_last_24h", 0.0)
    days_dry = weather_features.get("days_since_last_rain", 3)

    soil_moisture = np.clip(
        0.4 * (1 - days_dry / 30) + 0.3 * (precip / 50) + 0.3 * (humidity / 100), 0, 1
    )
    surface_runoff = precip * 0.05 * (1 + slope / 45)
    streamflow_index = np.clip(
        0.5 * (precip / 50) + 0.3 * (1 - elevation / 200.0) + 0.2 * soil_moisture, 0, 1
    )
    drought_index = np.clip(
        0.4 * (temp - 40) / 60 + 0.4 * (1 - humidity / 100) + 0.2 * (days_dry / 30), 0, 1
    )

    other = {
        "elevation": float(elevation),
        "slope": float(slope),
        "aspect": float(aspect),
        "distance_to_water": float(distance_to_water),
        "population_density": float(pop_density),
        "urbanization_index": float(urban_index),
        "impervious_surface_ratio": float(imperv),
        "vegetation_density": float(veg),
        "soil_moisture": float(soil_moisture),
        "surface_runoff": float(surface_runoff),
        "streamflow_index": float(streamflow_index),
        "drought_index": float(drought_index),
        "soil_type": "Sandy-Loam",
        "storm_warning_flag": 1 if weather_features.get("rainy_hours", 0) > 0 else 0,
        "recent_fire_flag": 0
    }
    return other

def make_fire_summary(town, fire_score, features, weather_features):
    """
    Generate a human-friendly fire hazard summary with suggested actions.
    """
    summary = {"warnings": [], "context": "", "user_actions": [], "government_actions": [], "severity": "LOW"}

    temp = features.get("temperature", 70)
    humidity = features.get("humidity", 60)
    wind = features.get("wind_speed", 5)
    drought_idx = features.get("drought_index", 0.3)
    veg_density = features.get("vegetation_density", 0.5)
    days_dry = features.get("days_since_last_rain", 3)

    if fire_score > 0.85:
        summary["severity"] = "EXTREME"
        summary["warnings"].append("EXTREME fire risk - Critical conditions for rapid spread")
    elif fire_score > 0.7:
        summary["severity"] = "HIGH"
        summary["warnings"].append("High fire risk - Dangerous for outdoor burning")
    elif fire_score > 0.4:
        summary["severity"] = "MODERATE"
        summary["warnings"].append("Moderate fire risk - Exercise caution")
    else:
        summary["severity"] = "LOW"
        summary["warnings"].append("Low fire risk - Conditions look favorable")

    # Build a readable context paragraph
    parts = [f"Fire Risk Analysis for {town}:"]
    if temp > 95:
        parts.append(f"Very hot ({temp}°F), which raises the chance of ignition and fast spread.")
    elif temp > 85:
        parts.append(f"Unusually warm ({temp}°F) and drying fuels.")
    elif temp > 75:
        parts.append(f"Warm ({temp}°F) — keep an eye on conditions.")
    else:
        parts.append(f"Temperatures ({temp}°F) are comfortable and not a major driver right now.")

    if humidity < 20:
        parts.append(f"Humidity is critically low ({humidity}%) — fuels are tinder-dry.")
    elif humidity < 35:
        parts.append(f"Low humidity ({humidity}%) increases fire potential.")
    elif humidity < 50:
        parts.append(f"Moderate humidity ({humidity}%) offers some natural suppression.")
    else:
        parts.append(f"Humidity ({humidity}%) is healthy and helps limit fires.")

    if wind > 25:
        parts.append(f"High winds ({wind} mph) could spread embers far from the ignition point.")
    elif wind > 15:
        parts.append(f"Moderate winds ({wind} mph) may accelerate fire spread.")
    else:
        parts.append(f"Light winds ({wind} mph) keep fire spread slower.")

    if drought_idx > 0.7 and veg_density > 0.5:
        parts.append(f"Dry conditions and dense vegetation mean there is abundant fuel for fires.")
    elif drought_idx > 0.5:
        parts.append(f"Moderate drought conditions after {days_dry} dry days.")

    summary["context"] = " ".join(parts)

    # Action lists scaled by severity
    if fire_score > 0.7:
        summary["user_actions"] = [
            "No outdoor burning — cancel campfires and burn permits",
            "Clear dry brush 30 feet from structures",
            "Water vulnerable landscaping",
            "Sign up for local emergency alerts",
            "Prepare an evacuation kit and plan"
        ]
        summary["government_actions"] = [
            "Issue immediate burn bans and public alerts",
            "Stage firefighting resources in high-risk zones",
            "Increase patrols and inspections",
            "Deploy public messaging on evacuation and safety",
            "Coordinate vegetation management near infrastructure"
        ]
    elif fire_score > 0.4:
        summary["user_actions"] = [
            "Delay non-essential outdoor fires",
            "Clean gutters and remove dry debris",
            "Supervise grills and smoking carefully",
            "Secure loose materials that can catch embers",
            "Water lawns and flammable plants"
        ]
        summary["government_actions"] = [
            "Increase monitoring and public education",
            "Schedule inspections for fire-prone sites",
            "Prepare resources for rapid deployment",
            "Issue guidance on defensible space",
            "Coordinate with local agencies for situational awareness"
        ]
    else:
        summary["user_actions"] = [
            "Maintain safe practices with open flames",
            "Regularly remove dry leaf litter",
            "Learn local evacuation routes",
            "Stay informed during dry spells"
        ]
        summary["government_actions"] = [
            "Continue routine inspections and maintenance",
            "Promote community fire safety programs",
            "Maintain fire access routes and water systems"
        ]

    return summary

def make_flood_summary(town, flood_score, features, weather_features, other):
    """
    Create a flood risk summary and useful actions in plain language.
    """
    summary = {"warnings": [], "context": "", "user_actions": [], "government_actions": [], "severity": "LOW"}

    precip24 = features.get("precipitation_last_24h", 0)
    soil_moisture = features.get("soil_moisture", 0.5)
    elev = features.get("elevation", 20)
    dist_water = features.get("distance_to_water", 5)
    imperv = features.get("impervious_surface_ratio", 0.4)
    streamflow = features.get("streamflow_index", 0.3)

    if flood_score > 0.85:
        summary["severity"] = "EXTREME"
        summary["warnings"].append("EXTREME flood risk - life-threatening flooding possible")
    elif flood_score > 0.7:
        summary["severity"] = "HIGH"
        summary["warnings"].append("High flood risk - significant flooding expected")
    elif flood_score > 0.4:
        summary["severity"] = "MODERATE"
        summary["warnings"].append("Moderate flood risk - localized flooding possible")
    else:
        summary["severity"] = "LOW"
        summary["warnings"].append("Low flood risk - normal drainage")

    parts = [f"Flood Risk Analysis for {town}:"]
    if precip24 > 40:
        parts.append(f"Extreme rainfall ({precip24:.1f}mm in 24 hours) may overwhelm drainage.")
    elif precip24 > 25:
        parts.append(f"Heavy rainfall ({precip24:.1f}mm) creating significant runoff.")
    elif precip24 > 10:
        parts.append(f"Moderate rainfall ({precip24:.1f}mm) is elevating water levels.")
    else:
        parts.append(f"Little recent rain ({precip24:.1f}mm) — flood risk low from rainfall.")

    if soil_moisture > 0.8:
        parts.append("Soil is saturated and cannot absorb more water.")
    elif soil_moisture > 0.6:
        parts.append("Soil moisture is high with limited absorption left.")
    elif soil_moisture > 0.4:
        parts.append("Soil moisture moderate, can absorb some rain.")
    else:
        parts.append("Soil has good absorption capacity.")

    if elev < 10 and dist_water < 2:
        parts.append("Low elevation and close to water increase coastal/tidal flood risk.")
    elif elev < 20:
        parts.append("Relatively low elevation increases pooling risk.")

    if imperv > 0.6:
        parts.append("High urban surfaces prevent natural drainage; water will run off.")
    elif imperv > 0.4:
        parts.append("Moderate urbanization affects drainage patterns.")

    if streamflow > 0.7:
        parts.append("Local waterways are at or near capacity.")
    elif streamflow > 0.5:
        parts.append("Drainage systems are handling current flow but are approaching limits.")

    summary["context"] = " ".join(parts)

    if flood_score > 0.7:
        summary["user_actions"] = [
            "Do not drive through flooded roads — Turn Around, Don't Drown",
            "Move to higher ground and evacuate low areas",
            "Elevate electronics and valuable items",
            "Turn off utilities if water reaches living spaces",
            "Keep emergency communication devices charged"
        ]
        summary["government_actions"] = [
            "Activate emergency operations and rescue teams",
            "Close flooded roads and deploy signage",
            "Stage swift-water rescue resources",
            "Issue wide-area emergency notifications",
            "Deploy pumps and flood-control hardware"
        ]
    elif flood_score > 0.4:
        summary["user_actions"] = [
            "Avoid known flood-prone spots and underpasses",
            "Clear gutters and ensure downspouts drain away from foundations",
            "Move vehicles to higher ground",
            "Clear yard drains and culverts",
            "Test sump pumps and battery backups"
        ]
        summary["government_actions"] = [
            "Inspect and clean storm drains",
            "Pre-position pumps and barricades",
            "Enhance rainfall and stream gauge monitoring",
            "Issue flood watches through local media",
            "Inspect critical infrastructure"
        ]
    else:
        summary["user_actions"] = [
            "Keep gutters clear and know your local flood risk",
            "Consider flood insurance if you're in a vulnerable area",
            "Maintain yard drains and be prepared with a small emergency kit"
        ]
        summary["government_actions"] = [
            "Routine storm drain maintenance",
            "Promote green infrastructure like rain gardens",
            "Enforce proper drainage in development planning",
            "Educate the public on flood preparedness"
        ]

    return summary

def make_activity_advice(fire_risk, flood_risk, features):
    """
    Recommend outdoor activities classified as safe / caution / not recommended.
    """
    temp = features.get("temperature", 70)
    wind = features.get("wind_speed", 5)
    precip = features.get("precipitation_last_24h", 0)
    humidity = features.get("humidity", 60)

    advice = {"safe": [], "caution": [], "not_recommended": []}

    # Hiking
    if fire_risk < 0.4 and flood_risk < 0.4 and temp < 85:
        advice["safe"].append("Hiking")
    elif fire_risk > 0.7 or flood_risk > 0.7:
        advice["not_recommended"].append("Hiking")
    else:
        advice["caution"].append("Hiking - Stay on marked trails")

    # Beach & Swimming
    if wind < 15 and 70 < temp < 95 and fire_risk < 0.6:
        advice["safe"].append("Beach & Swimming")
    elif wind > 25 or temp < 60 or temp > 100:
        advice["not_recommended"].append("Beach & Swimming")
    else:
        advice["caution"].append("Beach - Strong currents possible")

    # Camping
    if fire_risk < 0.3 and flood_risk < 0.3:
        advice["safe"].append("Camping")
    elif fire_risk > 0.6 or flood_risk > 0.6:
        advice["not_recommended"].append("Camping")
    else:
        advice["caution"].append("Camping - Check fire restrictions")

    # Cycling
    if wind < 20 and precip < 5 and temp < 90:
        advice["safe"].append("Cycling")
    elif wind > 30 or precip > 15:
        advice["not_recommended"].append("Cycling")
    else:
        advice["caution"].append("Cycling - Watch for wet roads")

    # Grilling
    if fire_risk < 0.4 and wind < 15:
        advice["safe"].append("Outdoor Grilling")
    elif fire_risk > 0.6:
        advice["not_recommended"].append("Outdoor Grilling")
    else:
        advice["caution"].append("Grilling - Use extreme caution")

    # Gardening
    if temp < 90 and fire_risk < 0.7:
        advice["safe"].append("Gardening")
    elif temp > 100:
        advice["caution"].append("Gardening - Hydrate frequently")
    else:
        advice["caution"].append("Gardening - Avoid peak sun hours")

    # Fishing
    if flood_risk < 0.5 and wind < 20:
        advice["safe"].append("Fishing")
    elif flood_risk > 0.7:
        advice["not_recommended"].append("Fishing")
    else:
        advice["caution"].append("Fishing - Monitor water levels")

    # Running
    if temp < 85 and fire_risk < 0.6 and humidity < 70:
        advice["safe"].append("Running & Jogging")
    elif temp > 95 or humidity > 85:
        advice["caution"].append("Running - Early morning/evening only")
    else:
        advice["caution"].append("Running - Stay hydrated")

    # Picnicking
    if fire_risk < 0.5 and flood_risk < 0.5 and 60 < temp < 90:
        advice["safe"].append("Picnicking")
    elif fire_risk > 0.7 or flood_risk > 0.7:
        advice["not_recommended"].append("Picnicking")
    else:
        advice["caution"].append("Picnicking - Choose location carefully")

    # Boating
    if wind < 15 and flood_risk < 0.5:
        advice["safe"].append("Boating")
    elif wind > 25 or flood_risk > 0.7:
        advice["not_recommended"].append("Boating")
    else:
        advice["caution"].append("Boating - Wear life jackets")

    # Photography and birding
    if fire_risk < 0.6 and flood_risk < 0.6:
        advice["safe"].append("Nature Photography")
    elif fire_risk > 0.8 or flood_risk > 0.8:
        advice["not_recommended"].append("Nature Photography")
    else:
        advice["caution"].append("Photography - Stay on safe paths")

    if temp < 90 and fire_risk < 0.7:
        advice["safe"].append("Bird Watching")
    else:
        advice["caution"].append("Bird Watching - Bring water")

    return advice

def prepare_features_for_model(all_features):
    """
    Turn our features dict into a numpy array suitable for the ML model,
    using the loaded scaler and one-hot encoder when available.
    """
    global std_scaler, onehot_encoder, feat_names, numeric_features, categorical_features

    if feat_names:
        if onehot_encoder is not None and hasattr(onehot_encoder, "feature_names_in_"):
            cat_cols_local = list(onehot_encoder.feature_names_in_())
            try:
                ohe_out_cols = list(onehot_encoder.get_feature_names_out())
            except Exception:
                ohe_out_cols = []
            num_cols_local = [fn for fn in feat_names if fn not in ohe_out_cols]
        else:
            num_cols_local = [
                'latitude', 'longitude', 'population_density', 'urbanization_index',
                'temperature', 'humidity', 'precipitation_last_24h', 'precipitation_last_7d',
                'wind_speed', 'days_since_last_rain', 'avg_temp_past_week', 'max_temp_past_week',
                'elevation', 'slope', 'aspect', 'distance_to_water', 'impervious_surface_ratio',
                'vegetation_density', 'soil_moisture', 'surface_runoff',
                'streamflow_index', 'drought_index', 'storm_warning_flag', 'recent_fire_flag'
            ]
            cat_cols_local = ['soil_type']
    else:
        num_cols_local = [
            'latitude', 'longitude', 'population_density', 'urbanization_index',
            'temperature', 'humidity', 'precipitation_last_24h', 'precipitation_last_7d',
            'wind_speed', 'days_since_last_rain', 'avg_temp_past_week', 'max_temp_past_week',
            'elevation', 'slope', 'aspect', 'distance_to_water', 'impervious_surface_ratio',
            'vegetation_density', 'soil_moisture', 'surface_runoff',
            'streamflow_index', 'drought_index', 'storm_warning_flag', 'recent_fire_flag'
        ]
        cat_cols_local = ['soil_type']

    row = {}
    for c in num_cols_local:
        row[c] = all_features.get(c, 0.0)
    for c in cat_cols_local:
        row[c] = all_features.get(c, "MISSING")

    X_num = np.array([[float(row[c]) for c in num_cols_local]])
    if cat_cols_local and onehot_encoder is not None:
        try:
            X_cat = onehot_encoder.transform(pd.DataFrame([{c: row[c] for c in cat_cols_local}]))
            X = np.hstack([X_num, X_cat])
        except Exception as exc:
            logger.warning("One-hot transform failed: %s", exc)
            X = X_num
    else:
        X = X_num

    if std_scaler is not None:
        try:
            X = std_scaler.transform(X)
        except Exception as exc:
            logger.warning("Scaler transform failed: %s", exc)

    return X, num_cols_local, cat_cols_local

def simple_rule_risks(features):
    """
    A fallback heuristic to estimate fire and flood risk when the ML model isn't available.
    It's intentionally simple and conservative.
    """
    t = features.get("temperature", 70)
    humid = features.get("humidity", 60)
    wind = features.get("wind_speed", 5)
    precip24 = features.get("precipitation_last_24h", 0)
    fire_risk = 0.0
    flood_risk = 0.0

    if t > 90: fire_risk += 0.25
    if t > 100: fire_risk += 0.25
    fire_risk += np.clip((60 - humid) / 60.0 * 0.3, 0, 0.3)
    fire_risk += np.clip((wind / 30.0) * 0.2, 0, 0.2)
    fire_risk = float(np.clip(fire_risk, 0, 1))

    flood_risk += np.clip(precip24 / 40.0, 0, 0.6)
    flood_risk += np.clip((1 - features.get("soil_moisture", 0.5)) * 0.2, 0, 0.2)
    flood_risk = float(np.clip(flood_risk, 0, 1))

    combined = max(fire_risk, flood_risk)
    if combined < 0.35:
        hazard = "low"
    elif combined < 0.65:
        hazard = "moderate"
    else:
        hazard = "high"

    return {"fire_risk": fire_risk, "flood_risk": flood_risk, "hazard_class": hazard}

def analyze_location(town):
    """
    Main orchestration for a single town:
    - fetch NOAA and elevation
    - derive features
    - call model (or rules) for scores
    - produce summaries and activity recommendations
    Returns (result_dict, error_or_none)
    """
    if town not in TOWNS:
        return None, "Unknown town"

    lat, lon = TOWNS[town]

    points = get_noaa_point(lat, lon)
    weather_feats, hourly_periods = extract_weather_features(lat, lon, points)

    elev = get_elevation(lat, lon)

    other = estimate_geo_features(lat, lon, elev, weather_feats)

    features = {
        "latitude": lat,
        "longitude": lon,
        "population_density": other["population_density"],
        "urbanization_index": other["urbanization_index"],
        "temperature": weather_feats["temperature"],
        "humidity": weather_feats["humidity"],
        "precipitation_last_24h": weather_feats["precipitation_last_24h"],
        "precipitation_last_7d": weather_feats["precipitation_last_7d"],
        "wind_speed": weather_feats["wind_speed"],
        "days_since_last_rain": weather_feats["days_since_last_rain"],
        "avg_temp_past_week": weather_feats["temperature"],
        "max_temp_past_week": weather_feats["temperature"] + 8.0,
        "elevation": other["elevation"],
        "slope": other["slope"],
        "aspect": other["aspect"],
        "distance_to_water": other["distance_to_water"],
        "impervious_surface_ratio": other["impervious_surface_ratio"],
        "vegetation_density": other["vegetation_density"],
        "soil_type": other["soil_type"],
        "soil_moisture": other["soil_moisture"],
        "surface_runoff": other["surface_runoff"],
        "streamflow_index": other["streamflow_index"],
        "drought_index": other["drought_index"],
        "storm_warning_flag": other["storm_warning_flag"],
        "recent_fire_flag": other["recent_fire_flag"]
    }

    # Build 24-hour arrays from NOAA hourly data if available
    hourly_temps = []
    hourly_winds = []
    if hourly_periods and len(hourly_periods) > 0:
        for period in hourly_periods[:24]:
            temp = period.get("temperature")
            if temp is not None:
                hourly_temps.append(float(temp))
            wind_str = period.get("windSpeed", "")
            m = re.search(r"(\d+)", str(wind_str))
            if m:
                hourly_winds.append(float(m.group(1)))
            else:
                hourly_winds.append(5.0)

    # Pad to 24 values
    cur_temp = features["temperature"]
    cur_wind = features["wind_speed"]
    while len(hourly_temps) < 24:
        hourly_temps.append(cur_temp)
    while len(hourly_winds) < 24:
        hourly_winds.append(cur_wind)
    hourly_temps = hourly_temps[:24]
    hourly_winds = hourly_winds[:24]

    ai_preds = [round(t, 1) for t in hourly_temps]
    wind_preds = [round(w, 1) for w in hourly_winds]
    confidence = 2.0

    # Use ML model when available, otherwise fallback to rules
    if ml_model is not None:
        try:
            X_in, used_nums, used_cats = prepare_features_for_model(features)
            preds = ml_model.predict(X_in, verbose=0)
            if isinstance(preds, list) and len(preds) >= 3:
                fire_pred = float(preds[0].reshape(-1)[0])
                flood_pred = float(preds[1].reshape(-1)[0])
                class_probs = preds[2].reshape(-1, preds[2].shape[-1])[0] if preds[2].ndim >= 2 else preds[2]
                class_idx = int(np.argmax(class_probs))
                if label_encoder is not None:
                    hazard_class = str(label_encoder.inverse_transform([class_idx])[0])
                else:
                    hazard_class = ["low", "moderate", "high"][class_idx] if class_idx < 3 else "moderate"
                used_model = True
            else:
                logger.warning("ML model returned unexpected output; using rules")
                r = simple_rule_risks(features)
                fire_pred = r["fire_risk"]
                flood_pred = r["flood_risk"]
                hazard_class = r["hazard_class"]
                used_model = False
        except Exception as exc:
            logger.warning("ML predict failed: %s", exc)
            r = simple_rule_risks(features)
            fire_pred = r["fire_risk"]
            flood_pred = r["flood_risk"]
            hazard_class = r["hazard_class"]
            used_model = False
    else:
        r = simple_rule_risks(features)
        fire_pred = r["fire_risk"]
        flood_pred = r["flood_risk"]
        hazard_class = r["hazard_class"]
        used_model = False

    # Summaries and recommendations
    fire_summary = make_fire_summary(town, fire_pred, features, weather_feats)
    flood_summary = make_flood_summary(town, flood_pred, features, weather_feats, other)
    activities = make_activity_advice(fire_pred, flood_pred, features)

    logger.info("Activities for %s: Safe=%s; Caution=%s; NotRecommended=%s", town, activities["safe"], activities["caution"], activities["not_recommended"])

    result = {
        "city": town,
        "lat": lat,
        "lon": lon,
        "forecast": (points.get("forecast").get("properties").get("periods")[0] if points and points.get("forecast") else None),
        "hourly_periods": hourly_periods,
        "ai_preds": ai_preds,
        "wind_preds": wind_preds,
        "confidence": confidence,
        "fire_risk": "HIGH" if fire_pred > 0.7 else "MODERATE" if fire_pred > 0.3 else "LOW",
        "fire_warnings": fire_summary["warnings"],
        "flood_risk": "HIGH" if flood_pred > 0.7 else "MODERATE" if flood_pred > 0.3 else "LOW",
        "flood_warnings": flood_summary["warnings"],
        "insights": [],
        "model_used": used_model,
        "raw_fire_score": float(fire_pred),
        "raw_flood_score": float(flood_pred),
        "fire_summary": fire_summary,
        "flood_summary": flood_summary,
        "activities": activities
    }

    # Quick human-friendly insight
    now_temp = features["temperature"]
    if now_temp < 40:
        result["insights"].append("Cold — wear a coat")
    elif now_temp > 80:
        result["insights"].append("Hot — stay hydrated")

    return result, None

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Web endpoint for the UI. On POST, runs analyze_location and renders the template.
    """
    town = None
    error_message = None
    lat = lon = None
    forecast = None
    hourly_periods = None
    ai_preds = None
    wind_preds = None
    confidence = None
    insights = []
    fire_risk = "LOW"
    fire_warnings = []
    flood_risk = "LOW"
    flood_warnings = []
    fire_summary = None
    flood_summary = None
    activities = None

    if request.method == "POST":
        town = request.form.get("city")
        if town:
            result, err = analyze_location(town)
            if err:
                error_message = err
            else:
                lat = result["lat"]
                lon = result["lon"]
                forecast = result["forecast"]
                hourly_periods = result["hourly_periods"]
                ai_preds = result["ai_preds"]
                wind_preds = result["wind_preds"]
                confidence = result["confidence"]
                insights = result["insights"]
                fire_risk = result["fire_risk"]
                fire_warnings = result["fire_warnings"]
                flood_risk = result["flood_risk"]
                flood_warnings = result["flood_warnings"]
                fire_summary = result["fire_summary"]
                flood_summary = result["flood_summary"]
                activities = result["activities"]

    return render_template("index.html",
        cities=sorted(TOWNS.keys()), city=town, lat=lat, lon=lon,
        forecast=forecast, ai_preds=ai_preds, wind_preds=wind_preds,
        confidence=confidence, insights=insights, error_message=error_message,
        fire_risk=fire_risk, fire_warnings=fire_warnings,
        flood_risk=flood_risk, flood_warnings=flood_warnings,
        fire_summary=fire_summary, flood_summary=flood_summary,
        activities=activities)

if __name__ == "__main__":
    print("="*60)
    print("Starting EnviroHazardLI")
    if MODEL_FILE.exists():
        print("Model file found:", MODEL_FILE)
    else:
        print("Model file NOT found. App will use rule-based fallback")
    if ENCODERS_FILE.exists():
        print("Encoders found:", ENCODERS_FILE)
    else:
        print("Encoders NOT found. Preprocessing may fallback.")
    print("Visit: http://localhost:5000")
    print("="*60)
    app.run(debug=True, host="0.0.0.0", port=5000)
