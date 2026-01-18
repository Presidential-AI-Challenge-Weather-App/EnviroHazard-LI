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
from flask import Flask, render_template_string, request, jsonify

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hazard_app")

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "hazard_multioutput.h5"
ENCODERS_PATH = MODEL_DIR / "scaler_and_encoders.pkl"
TOPO_MODEL_PATH = MODEL_DIR / "topography_risk.h5"
NOAA_API_KEY = ""

model = None
scaler = None
ohe = None
label_enc = None
feature_names = None
num_cols = []
cat_cols = []
topo_model = None
fire_model = None
flood_model = None

city_coords = {
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

weather_cache = {}
elevation_cache = {}
CACHE_TTL = 600

def load_model_and_encoders():
    global fire_model, flood_model, topo_model, scaler, ohe, label_enc, feature_names, num_cols, cat_cols

    fire_model, flood_model, topo_model = None, None, None
    scaler, ohe, label_enc, feature_names, num_cols, cat_cols = None, None, None, None, [], []

    # Load Fire model
    FIRE_MODEL_PATH = Path("fire_risk_model.h5")
    if FIRE_MODEL_PATH.exists():
        try:
            logger.info("Loading fire model from %s", FIRE_MODEL_PATH)
            fire_model = load_model(str(FIRE_MODEL_PATH), compile=False)
            logger.info("Fire model loaded successfully")
        except Exception as e:
            logger.warning("Failed to load fire model: %s", e)

    FLOOD_MODEL_PATH = Path("flood_risk_model.h5")
    if FLOOD_MODEL_PATH.exists():
        try:
            logger.info("Loading flood model from %s", FLOOD_MODEL_PATH)
            flood_model = load_model(str(FLOOD_MODEL_PATH), compile=False)
            logger.info("Flood model loaded successfully")
        except Exception as e:
            logger.warning("Failed to load flood model: %s", e)

    # Topography model
    if TOPO_MODEL_PATH.exists():
        try:
            logger.info("Loading topography model from %s", TOPO_MODEL_PATH)
            topo_model = load_model(str(TOPO_MODEL_PATH), compile=False)
            logger.info("Topography model loaded successfully")
        except Exception as e:
            logger.warning("Failed to load topography model: %s", e)
    else:
        logger.info("Topography model file not found")

    if ENCODERS_PATH.exists():
        try:
            with open(ENCODERS_PATH, "rb") as f:
                enc = pickle.load(f)
            scaler = enc.get("scaler")
            ohe = enc.get("ohe")
            label_enc = enc.get("label_enc")
            feature_names = enc.get("feature_names")
            num_cols = enc.get("num_cols", [])
            cat_cols = enc.get("cat_cols", [])
            logger.info("Encoders loaded. num_cols=%d cat_cols=%d", len(num_cols), len(cat_cols))
        except Exception as e:
            logger.warning("Failed to load encoders: %s", e)
    else:
        logger.info("Encoders file not found at %s", ENCODERS_PATH)
load_model_and_encoders()

def fetch_noaa_point(lat, lon):
    cache_key = f"noaa_{lat:.4f}_{lon:.4f}"
    if cache_key in weather_cache:
        data, ts = weather_cache[cache_key]
        if time.time() - ts < CACHE_TTL:
            return data

    headers = {"User-Agent": "(EnviroHazardLI, contact@example.com)"}
    try:
        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        r = requests.get(points_url, headers=headers, timeout=10)
        if r.status_code == 404:
            logger.info("NOAA: points 404 for %s,%s", lat, lon)
            return None
        r.raise_for_status()
        pdict = r.json().get("properties", {})
        forecast_url = pdict.get("forecast")
        hourly_url = pdict.get("forecastHourly")
        forecast = None
        hourly = None
        if forecast_url:
            time.sleep(0.35)
            r2 = requests.get(forecast_url, headers=headers, timeout=10)
            r2.raise_for_status()
            forecast = r2.json()
        if hourly_url:
            time.sleep(0.35)
            r3 = requests.get(hourly_url, headers=headers, timeout=10)
            r3.raise_for_status()
            hourly = r3.json()
        data = {"forecast": forecast, "hourly": hourly}
        weather_cache[cache_key] = (data, time.time())
        return data
    except Exception as e:
        logger.warning("NOAA fetch error: %s", e)
        return None

def fetch_usgs_elevation(lat, lon):
    cache_key = f"elev_{lat:.4f}_{lon:.4f}"
    if cache_key in elevation_cache:
        return elevation_cache[cache_key]
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
        elevation_cache[cache_key] = float(val)
        return float(val)
    except Exception as e:
        logger.warning("USGS elev fetch error: %s", e)
        return 20.0

def derive_features_from_noaa(lat, lon, points_json):
    default = {
        "temperature": 70.0, "humidity": 60.0, "wind_speed": 5.0,
        "precipitation_last_24h": 0.0, "precipitation_last_7d": 0.0, "days_since_last_rain": 3
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
            cur = periods[0]
            temperature = cur.get("temperature", temperature)
            wind_raw = cur.get("windSpeed", "")
            m = re.search(r"(\d+)", str(wind_raw))
            if m:
                wind_speed = float(m.group(1))
            dew = cur.get("dewpoint", {}).get("value")
            if dew is not None:
                dew_f = dew
                if dew < 60:
                    dew_f = dew * 9.0/5.0 + 32.0
                humidity = max(10.0, min(100.0, 100 - 5*(temperature - dew_f)))
            short_text = cur.get("shortForecast", "") if cur else ""
            detailed_text = cur.get("detailedForecast", "") if cur else ""
        else:
            short_text = ""
            detailed_text = ""

        rainy_hours = 0
        hourly_periods = hourly.get("properties", {}).get("periods", [])[:24] if hourly else []
        rain_keywords = ["rain", "shower", "storm", "thunder", "drizzle"]
        for h in hourly_periods:
            txt = (h.get("shortForecast","") + " " + h.get("detailedForecast","")).lower()
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
    except Exception as e:
        logger.warning("derive_features_from_noaa error: %s", e)
        return default, []

def estimate_other_features(lat, lon, elevation, features_noaa):
    slope = float(np.random.uniform(0, 6))
    aspect = float(np.random.uniform(0, 360))
    center_lon = -73.0
    distance_to_water = max(0.5, min(15.0, abs(lon - center_lon) * 50.0))

    urban_centers = [
        (40.7062, -73.6187),
        (40.8682, -73.4257)
    ]
    min_dist = min(np.sqrt((lat - ulat)**2 + (lon - ulon)**2) for ulat, ulon in urban_centers)
    if min_dist < 0.05:
        population_density = float(np.random.uniform(3000, 8000))
        urbanization_index = float(np.random.uniform(0.7, 0.9))
        impervious_surface_ratio = float(np.random.uniform(0.6, 0.9))
        vegetation_density = float(np.random.uniform(0.1, 0.3))
    elif min_dist < 0.15:
        population_density = float(np.random.uniform(1000, 3000))
        urbanization_index = float(np.random.uniform(0.4, 0.7))
        impervious_surface_ratio = float(np.random.uniform(0.3, 0.6))
        vegetation_density = float(np.random.uniform(0.3, 0.6))
    else:
        population_density = float(np.random.uniform(100, 1000))
        urbanization_index = float(np.random.uniform(0.1, 0.4))
        impervious_surface_ratio = float(np.random.uniform(0.1, 0.3))
        vegetation_density = float(np.random.uniform(0.6, 0.9))

    temp = features_noaa.get("temperature", 70.0)
    humidity = features_noaa.get("humidity", 60.0)
    precip = features_noaa.get("precipitation_last_24h", 0.0)
    days_dry = features_noaa.get("days_since_last_rain", 3)

    soil_moisture = np.clip(0.4 * (1 - days_dry / 30) + 0.3 * (precip / 50) + 0.3 * (humidity / 100), 0, 1)
    surface_runoff = precip * 0.05 * (1 + slope / 45)
    streamflow_index = np.clip(0.5 * (precip / 50) + 0.3 * (1 - elevation / 200.0) + 0.2 * soil_moisture, 0, 1)
    drought_index = np.clip(0.4 * (temp - 40) / 60 + 0.4 * (1 - humidity / 100) + 0.2 * (days_dry / 30), 0, 1)

    other = {
        "elevation": float(elevation),
        "slope": float(slope),
        "aspect": float(aspect),
        "distance_to_water": float(distance_to_water),
        "population_density": float(population_density),
        "urbanization_index": float(urbanization_index),
        "impervious_surface_ratio": float(impervious_surface_ratio),
        "vegetation_density": float(vegetation_density),
        "soil_moisture": float(soil_moisture),
        "surface_runoff": float(surface_runoff),
        "streamflow_index": float(streamflow_index),
        "drought_index": float(drought_index),
        "soil_type": "Sandy-Loam",
        "storm_warning_flag": 1 if features_noaa.get("rainy_hours", 0) > 0 else 0,
        "recent_fire_flag": 0
    }
    return other

def calculate_topography_risk(elevation, slope, aspect, distance_to_water, soil_type="Sandy-Loam"):
    global topo_model
    
    if topo_model is None:
        landslide_risk = min(1.0, slope / 30.0)
        erosion_risk = min(1.0, (slope / 25.0) * 0.7 + 0.3)
        
        coastal_risk = 0.0
        if elevation < 10 and distance_to_water < 2:
            coastal_risk = 0.9
        elif elevation < 20 and distance_to_water < 5:
            coastal_risk = 0.5
        elif elevation < 30:
            coastal_risk = 0.2
        
        drainage_risk = max(0.0, 1.0 - (elevation / 50.0))
        overall_risk = (landslide_risk * 0.25 + erosion_risk * 0.25 + coastal_risk * 0.3 + drainage_risk * 0.2)
        
        return {
            "overall_risk": float(overall_risk),
            "landslide_risk": float(landslide_risk),
            "erosion_risk": float(erosion_risk),
            "coastal_flood_risk": float(coastal_risk),
            "drainage_risk": float(drainage_risk),
            "severity": "HIGH" if overall_risk > 0.7 else "MODERATE" if overall_risk > 0.4 else "LOW"
        }
    
    try:
        X = np.array([[elevation, slope, aspect, distance_to_water]])
        prediction = topo_model.predict(X, verbose=0)
        overall_risk = float(prediction[0][0])
        
        return {
            "overall_risk": overall_risk,
            "landslide_risk": overall_risk * 0.8,
            "erosion_risk": overall_risk * 0.9,
            "coastal_flood_risk": overall_risk * 0.7,
            "drainage_risk": overall_risk * 0.6,
            "severity": "HIGH" if overall_risk > 0.7 else "MODERATE" if overall_risk > 0.4 else "LOW"
        }
    except Exception as e:
        logger.warning("Topography model prediction failed: %s", e)
        return calculate_topography_risk(elevation, slope, aspect, distance_to_water, soil_type)

def rule_based_risks(features):
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


def generate_fire_summary(city, fire_risk_score, features, noaa_features):
    summary = {
        "warnings": [],
        "context": "",
        "user_actions": [],
        "government_actions": [],
        "severity": "LOW"
    }
    
    temp = features.get("temperature", 70)
    humidity = features.get("humidity", 60)
    wind = features.get("wind_speed", 5)
    drought_idx = features.get("drought_index", 0.3)
    veg_density = features.get("vegetation_density", 0.5)
    days_dry = features.get("days_since_last_rain", 3)
    
    if fire_risk_score > 0.85:
        summary["severity"] = "EXTREME"
        summary["warnings"].append("EXTREME fire risk - Critical fire weather conditions")
    elif fire_risk_score > 0.7:
        summary["severity"] = "HIGH"
        summary["warnings"].append("High fire risk - Dangerous conditions for fire spread")
    elif fire_risk_score > 0.4:
        summary["severity"] = "MODERATE"
        summary["warnings"].append("Moderate fire risk - Be cautious with outdoor activities")
    else:
        summary["severity"] = "LOW"
        summary["warnings"].append("Low fire risk - Conditions are favorable")
    
    context_parts = [f"Fire Risk Analysis for {city}:"]
    
    if temp > 95:
        context_parts.append(f"Extreme heat ({temp}°F) is creating severe fire conditions.")
    elif temp > 85:
        context_parts.append(f"High temperatures ({temp}°F) are elevating fire danger.")
    elif temp > 75:
        context_parts.append(f"Warm temperatures ({temp}°F) contribute to moderate fire risk.")
    else:
        context_parts.append(f"Current temperatures ({temp}°F) are relatively safe.")
    
    if humidity < 20:
        context_parts.append(f"Critically low humidity ({humidity}%) creates tinder-dry conditions where fires can ignite easily and spread rapidly.")
    elif humidity < 35:
        context_parts.append(f"Low humidity ({humidity}%) increases fuel flammability and fire spread potential.")
    elif humidity < 50:
        context_parts.append(f"Moderate humidity ({humidity}%) provides some natural fire suppression.")
    else:
        context_parts.append(f"Good humidity levels ({humidity}%) help prevent fire ignition.")
    
    if wind > 25:
        context_parts.append(f"Strong winds ({wind} mph) can carry embers up to a mile, rapidly spreading fires across neighborhoods.")
    elif wind > 15:
        context_parts.append(f"Moderate winds ({wind} mph) could accelerate fire spread if ignition occurs.")
    else:
        context_parts.append(f"Light winds ({wind} mph) limit fire spread capability.")
    
    if drought_idx > 0.7 and veg_density > 0.5:
        context_parts.append(f"Prolonged dry conditions ({days_dry} days since rain) combined with dense vegetation create abundant fuel for potential fires.")
    elif drought_idx > 0.5:
        context_parts.append(f"Dry spell of {days_dry} days has reduced moisture in vegetation and soil.")
    
    summary["context"] = " ".join(context_parts)
    
    if fire_risk_score > 0.7:
        summary["user_actions"] = [
            "NO outdoor burning - Cancel all planned campfires, barbecues, and yard waste burning",
            "Create defensible space - Clear dry leaves, branches, and debris within 30 feet of structures",
            "Water exposed areas - Keep lawns and vegetation near buildings well-watered",
            "Stay informed - Sign up for local emergency alerts (NY-Alert system)",
            "Prepare evacuation kit - Have essentials ready: documents, medications, water, flashlight"
        ]
        summary["government_actions"] = [
            "Issue burn bans - Prohibit all outdoor burning permits and fireworks",
            "Increase fire patrols - Deploy additional fire marshals to high-risk areas",
            "Pre-position fire resources - Stage fire trucks and crews in vulnerable neighborhoods",
            "Public awareness campaign - Activate emergency notification systems and social media alerts",
            "Emergency vegetation management - Clear fire breaks along critical infrastructure"
        ]
    elif fire_risk_score > 0.4:
        summary["user_actions"] = [
            "Limit outdoor burning - Postpone non-essential fires; use fire pits with screens",
            "Clean gutters and roofs - Remove dry leaves and pine needles",
            "Supervise all flames - Never leave fires, grills, or smoking materials unattended",
            "Secure loose materials - Bring in patio furniture cushions and secure trash cans",
            "Water landscaping - Maintain healthy, moist vegetation near your home"
        ]
        summary["government_actions"] = [
            "Monitor conditions - Increase weather monitoring and fire danger assessments",
            "Public education - Distribute fire safety information through community centers",
            "Inspect fire hydrants - Verify water supply systems are operational",
            "Enforce codes - Ensure commercial properties maintain defensible space",
            "Enhance communication - Update emergency contact databases and test alert systems"
        ]
    else:
        summary["user_actions"] = [
            "Maintain vigilance - Continue safe practices with outdoor fires",
            "Regular maintenance - Keep property clear of excessive dry vegetation",
            "Practice fire safety - Always extinguish campfires and cigarettes completely",
            "Stay educated - Learn about local fire risks and evacuation routes"
        ]
        summary["government_actions"] = [
            "Routine inspections - Continue regular fire code enforcement",
            "Vegetation management - Maintain scheduled brush clearing along roadways",
            "Community training - Offer fire safety workshops and CERT programs",
            "Infrastructure maintenance - Keep fire access roads and water systems in good repair"
        ]
    
    return summary

def generate_flood_summary(city, flood_risk_score, features, noaa_features, other):
    summary = {
        "warnings": [],
        "context": "",
        "user_actions": [],
        "government_actions": [],
        "severity": "LOW"
    }
    
    precip24 = features.get("precipitation_last_24h", 0)
    precip7 = features.get("precipitation_last_7d", 0)
    soil_moisture = features.get("soil_moisture", 0.5)
    elev = features.get("elevation", 20)
    dist_water = features.get("distance_to_water", 5)
    impervious = features.get("impervious_surface_ratio", 0.4)
    streamflow = features.get("streamflow_index", 0.3)
    
    if flood_risk_score > 0.85:
        summary["severity"] = "EXTREME"
        summary["warnings"].append("EXTREME flood risk - Life-threatening flooding possible")
    elif flood_risk_score > 0.7:
        summary["severity"] = "HIGH"
        summary["warnings"].append("High flood risk - Significant flooding expected")
    elif flood_risk_score > 0.4:
        summary["severity"] = "MODERATE"
        summary["warnings"].append("Moderate flood risk - Localized flooding possible")
    else:
        summary["severity"] = "LOW"
        summary["warnings"].append("Low flood risk - Normal drainage conditions")
    
    context_parts = [f"Flood Risk Analysis for {city}:"]
    
    if precip24 > 40:
        context_parts.append(f"Extreme rainfall ({precip24:.1f}mm in 24hrs) has saturated the ground and overwhelmed drainage systems.")
    elif precip24 > 25:
        context_parts.append(f"Heavy rainfall ({precip24:.1f}mm in 24hrs) is creating significant runoff and drainage challenges.")
    elif precip24 > 10:
        context_parts.append(f"Moderate rainfall ({precip24:.1f}mm in 24hrs) is contributing to elevated water levels.")
    else:
        context_parts.append(f"Light to no recent rainfall ({precip24:.1f}mm in 24hrs) keeps flood risk minimal.")
    
    if soil_moisture > 0.8:
        context_parts.append(f"Soil is completely saturated ({soil_moisture*100:.0f}% capacity) - additional rainfall cannot be absorbed and will become immediate runoff.")
    elif soil_moisture > 0.6:
        context_parts.append(f"Soil moisture is high ({soil_moisture*100:.0f}% capacity) with limited absorption capacity remaining.")
    elif soil_moisture > 0.4:
        context_parts.append(f"Soil moisture at moderate levels ({soil_moisture*100:.0f}% capacity) can handle some additional rain.")
    else:
        context_parts.append(f"Soil has good absorption capacity ({soil_moisture*100:.0f}% capacity) to handle rainfall.")
    
    if elev < 10 and dist_water < 2:
        context_parts.append(f"Low elevation ({elev:.0f}m) and proximity to water bodies create high vulnerability to coastal and tidal flooding.")
    elif elev < 20:
        context_parts.append(f"Relatively low elevation ({elev:.0f}m) increases susceptibility to pooling water.")
    
    if impervious > 0.6:
        context_parts.append(f"High urban development ({impervious*100:.0f}% impervious surfaces) prevents natural drainage - water runs off pavement instead of soaking into ground.")
    elif impervious > 0.4:
        context_parts.append(f"Moderate urbanization ({impervious*100:.0f}% impervious) affects natural drainage patterns.")
    
    if streamflow > 0.7:
        context_parts.append("Local waterways and storm drains are at or near capacity - any additional rainfall may cause overflow.")
    elif streamflow > 0.5:
        context_parts.append("Drainage systems are handling current volume but approaching limits.")
    
    summary["context"] = " ".join(context_parts)
    
    if flood_risk_score > 0.7:
        summary["user_actions"] = [
            "NEVER drive through flooded roads - Turn Around, Don't Drown! Just 6 inches can stall your vehicle",
            "Move to higher ground - Evacuate basements and low-lying areas immediately",
            "Protect electronics - Elevate computers, TVs, and appliances off the floor",
            "Shut off utilities if flooding - Turn off electricity and gas if water enters your home",
            "Emergency communication - Keep phone charged; text instead of call to save battery"
        ]
        summary["government_actions"] = [
            "Activate Emergency Operations Center - Coordinate multi-agency flood response",
            "Close flooded roads - Barricade dangerous crossings; deploy signage and personnel",
            "Position rescue teams - Stage swift-water rescue units in vulnerable areas",
            "Emergency notifications - Send Wireless Emergency Alerts to all cell phones in affected zones",
            "Deploy flood pumps - Use portable pumps to clear streets and critical infrastructure"
        ]
    elif flood_risk_score > 0.4:
        summary["user_actions"] = [
            "Avoid flood-prone areas - Stay away from known low spots, underpasses, and creek crossings",
            "Clear gutters and drains - Ensure downspouts direct water away from foundation",
            "Move vehicles to higher ground - Don't park in low-lying areas or near storm drains",
            "Prepare drainage - Clear yard drains, culverts, and ditches of debris",
            "Check sump pump - Test battery backup; ensure discharge pipe is clear"
        ]
        summary["government_actions"] = [
            "Monitor storm drains - Inspect and clear critical drainage infrastructure",
            "Pre-position equipment - Stage pumps, barricades, and emergency supplies",
            "Weather monitoring - Enhance rainfall tracking and stream gauge observation",
            "Issue flood watches - Alert public through social media and local news",
            "Inspect infrastructure - Check bridges, culverts, and retention ponds"
        ]
    else:
        summary["user_actions"] = [
            "Maintain drainage - Keep gutters and downspouts clear year-round",
            "Know your risk - Understand if you're in a flood zone",
            "Consider flood insurance - Standard homeowners insurance doesn't cover floods",
            "Regular maintenance - Keep yard drains and French drains functional",
            "Be prepared - Have an emergency kit ready even when risk is low"
        ]
        summary["government_actions"] = [
            "Routine maintenance - Regular storm drain cleaning and infrastructure inspections",
            "Green infrastructure - Develop rain gardens and permeable surfaces to absorb runoff",
            "Code enforcement - Ensure new development includes proper drainage planning",
            "Public education - Offer workshops on flood insurance and household preparedness",
            "Data collection - Maintain rain gauges and stream monitoring networks"
        ]
    return summary

def generate_activity_recommendations(fire_risk, flood_risk, features):
    temp = features.get("temperature", 70)
    wind = features.get("wind_speed", 5)
    precip = features.get("precipitation_last_24h", 0)
    humidity = features.get("humidity", 60)
    
    activities = {
        "safe": [],
        "caution": [],
        "not_recommended": []
    }
    
    if fire_risk < 0.4 and flood_risk < 0.4 and temp < 85:
        activities["safe"].append("Hiking")
    elif fire_risk > 0.7 or flood_risk > 0.7:
        activities["not_recommended"].append("Hiking")
    else:
        activities["caution"].append("Hiking - Stay on marked trails")
    
    if wind < 15 and temp > 70 and temp < 95 and fire_risk < 0.6:
        activities["safe"].append("Beach & Swimming")
    elif wind > 25 or temp < 60 or temp > 100:
        activities["not_recommended"].append("Beach & Swimming")
    else:
        activities["caution"].append("Beach - Strong currents possible")
    
    if fire_risk < 0.3 and flood_risk < 0.3:
        activities["safe"].append("Camping")
    elif fire_risk > 0.6 or flood_risk > 0.6:
        activities["not_recommended"].append("Camping")
    else:
        activities["caution"].append("Camping - Check fire restrictions")
    
    if wind < 20 and precip < 5 and temp < 90:
        activities["safe"].append("Cycling")
    elif wind > 30 or precip > 15:
        activities["not_recommended"].append("Cycling")
    else:
        activities["caution"].append("Cycling - Watch for wet roads")
    
    if fire_risk < 0.4 and wind < 15:
        activities["safe"].append("Outdoor Grilling")
    elif fire_risk > 0.6:
        activities["not_recommended"].append("Outdoor Grilling")
    else:
        activities["caution"].append("Grilling - Use extreme caution")
    
    if temp < 90 and fire_risk < 0.7:
        activities["safe"].append("Gardening")
    elif temp > 100:
        activities["caution"].append("Gardening - Hydrate frequently")
    else:
        activities["caution"].append("Gardening - Avoid peak sun hours")
    
    if flood_risk < 0.5 and wind < 20:
        activities["safe"].append("Fishing")
    elif flood_risk > 0.7:
        activities["not_recommended"].append("Fishing")
    else:
        activities["caution"].append("Fishing - Monitor water levels")
    
    if temp < 85 and fire_risk < 0.6 and humidity < 70:
        activities["safe"].append("Running & Jogging")
    elif temp > 95 or humidity > 85:
        activities["caution"].append("Running - Early morning/evening only")
    else:
        activities["caution"].append("Running - Stay hydrated")
    
    if fire_risk < 0.5 and flood_risk < 0.5 and temp > 60 and temp < 90:
        activities["safe"].append("Picnicking")
    elif fire_risk > 0.7 or flood_risk > 0.7:
        activities["not_recommended"].append("Picnicking")
    else:
        activities["caution"].append("Picnicking - Choose location carefully")
    
    if wind < 15 and flood_risk < 0.5:
        activities["safe"].append("Boating")
    elif wind > 25 or flood_risk > 0.7:
        activities["not_recommended"].append("Boating")
    else:
        activities["caution"].append("Boating - Wear life jackets")
    
    if fire_risk < 0.6 and flood_risk < 0.6:
        activities["safe"].append("Nature Photography")
    elif fire_risk > 0.8 or flood_risk > 0.8:
        activities["not_recommended"].append("Nature Photography")
    else:
        activities["caution"].append("Photography - Stay on safe paths")
    
    if temp < 90 and fire_risk < 0.7:
        activities["safe"].append("Bird Watching")
    else:
        activities["caution"].append("Bird Watching - Bring water")
    
    return activities

def preprocess_features_for_model(all_features):
    global scaler, ohe, feature_names, num_cols, cat_cols

    if feature_names:
        if ohe is not None and hasattr(ohe, "feature_names_in_"):
            cat_cols_local = list(ohe.feature_names_in_())
            try:
                ohe_out_cols = list(ohe.get_feature_names_out())
            except Exception:
                ohe_out_cols = []
            num_cols_local = [fn for fn in feature_names if fn not in ohe_out_cols]
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

    df_row = {}
    for c in num_cols_local:
        df_row[c] = all_features.get(c, 0.0)
    for c in cat_cols_local:
        df_row[c] = all_features.get(c, "MISSING")

    X_num = np.array([[float(df_row[c]) for c in num_cols_local]])
    if cat_cols_local and ohe is not None:
        try:
            X_cat = ohe.transform(pd.DataFrame([{c: df_row[c] for c in cat_cols_local}]))
            X = np.hstack([X_num, X_cat])
        except Exception as e:
            logger.warning("OHE transform failed: %s", e)
            X = X_num
    else:
        X = X_num

    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as e:
            logger.warning("Scaler transform failed: %s", e)

    return X, num_cols_local, cat_cols_local


def predict_for_city(city):
    if city not in city_coords:
        return None, "Unknown city"
    lat, lon = city_coords[city]
    points = fetch_noaa_point(lat, lon)
    noaa_features, hourly_periods = derive_features_from_noaa(lat, lon, points)
    elev = fetch_usgs_elevation(lat, lon)
    other = estimate_other_features(lat, lon, elev, noaa_features)
    features = {
        "latitude": lat, "longitude": lon,
        "population_density": other["population_density"],
        "urbanization_index": other["urbanization_index"],
        "temperature": noaa_features["temperature"],
        "humidity": noaa_features["humidity"],
        "precipitation_last_24h": noaa_features["precipitation_last_24h"],
        "precipitation_last_7d": noaa_features["precipitation_last_7d"],
        "wind_speed": noaa_features["wind_speed"],
        "days_since_last_rain": noaa_features["days_since_last_rain"],
        "avg_temp_past_week": noaa_features["temperature"],
        "max_temp_past_week": noaa_features["temperature"] + 8.0,
        "elevation": other["elevation"], "slope": other["slope"],
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
    ai_temps, ai_winds = [], []
    if hourly_periods:
        for period in hourly_periods[:24]:
            temp = period.get("temperature")
            if temp: ai_temps.append(float(temp))
            wind_str = period.get("windSpeed", "")
            wind_match = re.search(r"(\d+)", str(wind_str))
            ai_winds.append(float(wind_match.group(1)) if wind_match else 5.0)
    while len(ai_temps) < 24: ai_temps.append(features["temperature"])
    while len(ai_winds) < 24: ai_winds.append(features["wind_speed"])
    ai_preds = [round(t, 1) for t in ai_temps[:24]]
    wind_preds = [round(w, 1) for w in ai_winds[:24]]
    if model:
        try:
            X_input, _, _ = preprocess_features_for_model(features)
            preds = model.predict(X_input, verbose=0)
            if isinstance(preds, list) and len(preds) >= 3:
                fire_pred, flood_pred = float(preds[0].reshape(-1)[0]), float(preds[1].reshape(-1)[0])
            else:
                r = rule_based_risks(features)
                fire_pred, flood_pred = r["fire_risk"], r["flood_risk"]
        except:
            r = rule_based_risks(features)
            fire_pred, flood_pred = r["fire_risk"], r["flood_risk"]
    else:
        r = rule_based_risks(features)
        fire_pred, flood_pred = r["fire_risk"], r["flood_risk"]
    fire_summary = generate_fire_summary(city, fire_pred, features, noaa_features)
    flood_summary = generate_flood_summary(city, flood_pred, features, noaa_features, other)
    activities = generate_activity_recommendations(fire_pred, flood_pred, features)
    topo_risk = calculate_topography_risk(other["elevation"], other["slope"], other["aspect"], other["distance_to_water"], other["soil_type"])
    return {
        "city": city, "lat": lat, "lon": lon,
        "forecast": (points.get("forecast").get("properties").get("periods")[0] if points and points.get("forecast") else None),
        "ai_preds": ai_preds, "wind_preds": wind_preds,
        "fire_risk": "HIGH" if fire_pred > 0.7 else "MODERATE" if fire_pred > 0.3 else "LOW",
        "fire_warnings": fire_summary["warnings"],
        "flood_risk": "HIGH" if flood_pred > 0.7 else "MODERATE" if flood_pred > 0.3 else "LOW",
        "flood_warnings": flood_summary["warnings"],
        "raw_fire_score": float(fire_pred), "raw_flood_score": float(flood_pred),
        "fire_summary": fire_summary, "flood_summary": flood_summary,
        "activities": activities, "topography_risk": topo_risk
    }, None
def nn_predict_fire_flood(features_dict):
    try:
        X_input, _, _ = preprocess_features_for_model(features_dict)  # shape (1, n_features)

        if fire_model is None or flood_model is None:
            raise ValueError("NN model not loaded")

        fire_pred = float(fire_model.predict(X_input, verbose=0)[0][0])
        flood_pred = float(flood_model.predict(X_input, verbose=0)[0][0])
        return fire_pred, flood_pred

    except Exception as e:
        logger.warning("NN failed, using fallback: %s", e)
        r = rule_based_risks(features_dict)
        return r["fire_risk"], r["flood_risk"]
@app.route("/whatif", methods=["POST"])
def whatif_simulator():
    try:
        data = request.get_json()
        city = data.get("city")
        if not city or city not in city_coords:
            return jsonify({"error": "Invalid city"}), 400
        lat, lon = city_coords[city]
        points = fetch_noaa_point(lat, lon)
        noaa_features, _ = derive_features_from_noaa(lat, lon, points)
        elev = fetch_usgs_elevation(lat, lon)
        other = estimate_other_features(lat, lon, elev, noaa_features)
        base_features = {
            "temperature": noaa_features["temperature"], "humidity": noaa_features["humidity"],
            "precipitation_last_24h": noaa_features["precipitation_last_24h"],
            "wind_speed": noaa_features["wind_speed"],
            "soil_moisture": other["soil_moisture"], "drought_index": other["drought_index"]
        }
        orig_fire, orig_flood = nn_predict_fire_flood({ **base_features, "elevation": elev })
        
        sim_features = base_features.copy()
        if "temperature" in data: sim_features["temperature"] = float(data["temperature"])
        if "humidity" in data: sim_features["humidity"] = float(data["humidity"])
        if "wind_speed" in data: sim_features["wind_speed"] = float(data["wind_speed"])
        if "precipitation" in data: sim_features["precipitation_last_24h"] = float(data["precipitation"])
        temp, humidity, precip = sim_features["temperature"], sim_features["humidity"], sim_features["precipitation_last_24h"]
        days_dry = 0 if precip > 0 else 3
        sim_features["soil_moisture"] = float(np.clip(0.4 * (1 - days_dry / 30) + 0.3 * (precip / 50) + 0.3 * (humidity / 100), 0, 1))
        sim_features["drought_index"] = float(np.clip(0.4 * (temp - 40) / 60 + 0.4 * (1 - humidity / 100) + 0.2 * (days_dry / 30), 0, 1))
        sim_fire, sim_flood = nn_predict_fire_flood({ **sim_features, "elevation": elev })

        return jsonify({
    "original": {
        "fire_risk": orig_fire,
        "flood_risk": orig_flood
    },
    "simulated": {
        "fire_risk": sim_fire,
        "flood_risk": sim_flood
    }
})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET", "POST"])
def index():
    city, lat, lon, forecast, ai_preds, wind_preds = None, None, None, None, None, None
    fire_risk, flood_risk, fire_summary, flood_summary = "LOW", "LOW", None, None
    activities, topography_risk, error_message = None, None, None
    if request.method == "POST":
        city = request.form.get("city")
        if city:
            result, err = predict_for_city(city)
            if err:
                error_message = err
            else:
                lat, lon, forecast = result["lat"], result["lon"], result["forecast"]
                ai_preds, wind_preds = result["ai_preds"], result["wind_preds"]
                fire_risk, flood_risk = result["fire_risk"], result["flood_risk"]
                fire_summary, flood_summary = result["fire_summary"], result["flood_summary"]
                activities, topography_risk = result["activities"], result["topography_risk"]
    return render_template_string("""<!DOCTYPE html>
<html>
   <head>
      <title>EnviroHazardLI</title>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width,initial-scale=1.0">
      <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
      <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
      <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
      <style>*{box-sizing:border-box;margin:0;padding:0}body{font-family:'Inter','Segoe UI',Arial,sans-serif;background:#0a1f0a;color:#e8f5e9;min-height:100vh}.container{max-width:1400px;margin:0 auto;padding:20px}header{text-align:center;padding:25px 20px;background:linear-gradient(135deg,#1b5e20 0%,#2e7d32 100%);border-radius:12px;margin-bottom:25px;box-shadow:0 4px 6px rgba(0,0,0,0.3)}header h1{font-size:2.2em;font-weight:700;color:#a5d6a7;margin-bottom:5px}header p{color:#c8e6c9;font-size:0.95em}.form-container{background:#1b3a1b;padding:20px;border-radius:12px;margin-bottom:25px;box-shadow:0 4px 6px rgba(0,0,0,0.3);text-align:center;border:1px solid #2e7d32}select,button{padding:10px 18px;font-size:0.95em;border-radius:8px;border:1px solid #4caf50;margin:0 8px;background:#1b3a1b;color:#c8e6c9}button{background:linear-gradient(135deg,#2e7d32 0%,#388e3c 100%);color:#ffffff;cursor:pointer;font-weight:600;border:none;transition:all 0.3s ease}button:hover{background:linear-gradient(135deg,#388e3c 0%,#43a047 100%)}.main-content{display:flex;gap:20px;margin-bottom:25px}.map-container{flex:0 0 60%}#map{height:500px;width:100%;border-radius:12px;box-shadow:0 4px 6px rgba(0,0,0,0.3);border:2px solid #2e7d32}.weather-sidebar{flex:0 0 38%;background:#1b3a1b;border-radius:12px;padding:20px;box-shadow:0 4px 6px rgba(0,0,0,0.3);border:1px solid #2e7d32;overflow-y:auto;max-height:500px}.weather-sidebar h3{color:#a5d6a7;margin-bottom:15px;font-size:1.3em;border-bottom:2px solid #2e7d32;padding-bottom:8px}.weather-info{background:#0f2a0f;padding:12px;border-radius:8px;margin-bottom:12px;border-left:3px solid #4caf50}.weather-info strong{color:#81c784}canvas{width:100%!important;height:180px!important;margin-top:10px}.risk-card{background:#1b3a1b;border-radius:12px;padding:18px;margin:15px 0;box-shadow:0 4px 6px rgba(0,0,0,0.3);border-left:4px solid #4caf50}.risk-card.low{border-left-color:#66bb6a}.risk-card.moderate{border-left-color:#ffa726}.risk-card.high{border-left-color:#ef5350}.card-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #2e7d32}.card-header h3{color:#a5d6a7;font-size:1.2em;margin:0}.severity-badge{padding:5px 12px;border-radius:16px;font-weight:700;font-size:0.85em}.severity-badge.LOW{background:#1b5e20;color:#a5d6a7}.severity-badge.MODERATE{background:#e65100;color:#ffcc80}.severity-badge.HIGH{background:#b71c1c;color:#ffcdd2}.warning-box{background:rgba(255,167,38,0.1);padding:10px;border-radius:6px;margin:8px 0;border-left:3px solid #ffa726;font-weight:600;color:#ffcc80}.summary-section{margin-top:12px;padding:12px;background:#0f2a0f;border-radius:8px}.summary-section h4{color:#81c784;font-size:1em;margin-bottom:8px}.context-text{background:rgba(76,175,80,0.1);padding:10px;border-radius:6px;border-left:3px solid #4caf50;margin:8px 0;line-height:1.6;color:#c8e6c9;font-size:0.9em}.action-list{list-style:none;padding:0;margin:8px 0}.action-list li{padding:8px 10px;margin:6px 0;background:rgba(76,175,80,0.05);border-radius:6px;border-left:2px solid #4caf50;line-height:1.4;color:#c8e6c9;font-size:0.88em}input[type="range"]{-webkit-appearance:none;appearance:none;width:100%;height:8px;background:#2e7d32;border-radius:5px;outline:none;margin:5px 0}input[type="range"]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:20px;height:20px;background:#66bb6a;cursor:pointer;border-radius:50%}input[type="range"]::-moz-range-thumb{width:20px;height:20px;background:#66bb6a;cursor:pointer;border-radius:50%;border:none}.error{background:rgba(211,47,47,0.2);color:#ffcdd2;padding:12px;border-radius:8px;margin:15px 0;border-left:3px solid #d32f2f}</style>
   </head>
                                  <script>
function updateSliderValues() {
    document.getElementById("sim-temp-val").innerText = document.getElementById("sim-temp").value + "°F";
    document.getElementById("sim-humidity-val").innerText = document.getElementById("sim-humidity").value + "%";
    document.getElementById("sim-wind-val").innerText = document.getElementById("sim-wind").value + " mph";
    document.getElementById("sim-precip-val").innerText = document.getElementById("sim-precip").value + " mm";
}

// Update values when sliders move
document.getElementById("sim-temp").addEventListener("input", updateSliderValues);
document.getElementById("sim-humidity").addEventListener("input", updateSliderValues);
document.getElementById("sim-wind").addEventListener("input", updateSliderValues);
document.getElementById("sim-precip").addEventListener("input", updateSliderValues);

async function runSimulation() {
    const city = document.getElementById("city-select").value; // make sure you have a select box with id="city-select"
    const data = {
        city: city,
        temperature: document.getElementById("sim-temp").value,
        humidity: document.getElementById("sim-humidity").value,
        wind_speed: document.getElementById("sim-wind").value,
        precipitation: document.getElementById("sim-precip").value
    };

    const resultDiv = document.getElementById("simulation-results");
    resultDiv.style.display = "block";  // show the results div

    try {
        const response = await fetch("/whatif", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (result.error) {
            alert("Error: " + result.error);
            return;
        }

        document.getElementById("fire-change").innerText = 
            (result.simulated.fire_risk * 100).toFixed(0) + "%";
        document.getElementById("flood-change").innerText = 
            (result.simulated.flood_risk * 100).toFixed(0) + "%";

    } catch (err) {
        console.error(err);
        alert("Failed to run simulation.");
    }
}
</script>
   <body>
      <div class="container">
         <header>
            <h1>EnviroHazardLI</h1>
            <p>Environmental Hazard Monitoring for Long Island</p>
         </header>
         <div class="form-container">
            <form method="POST">
               <label for="city" style="font-weight:600;color:#a5d6a7;">Select Location: </label>
               <select name="city" id="city">
                  <option value="">Choose a town...</option>
                  {% for c in cities %}<option value="{{c}}" {% if c == city %}selected{% endif %}>{{c}}</option>{% endfor %}
               </select>
               <button type="submit">Analyze Hazards</button>
            </form>
         </div>
         {% if error_message %}
         <div class="error"><strong>Error:</strong>{{error_message}}</div>
         {% endif %}{% if city and lat %}
         <div class="main-content">
            <div class="map-container">
               <div id="map"></div>
            </div>
            <div class="weather-sidebar">
               <h3>{{city}}</h3>
               <div class="weather-info">
                  <div><strong>Coordinates:</strong> {{"%.4f"|format(lat)}}, {{"%.4f"|format(lon)}}</div>
               </div>
               {% if forecast %}
               <div class="weather-info"><strong>Current Conditions</strong><br>{{forecast['name']}} - {{forecast['shortForecast']}}<br><strong>Temperature:</strong> {{forecast['temperature']}}°{{forecast['temperatureUnit']}}<br>{% if forecast.get('windSpeed') %}<strong>Wind:</strong> {{forecast['windSpeed']}} {{forecast.get('windDirection','')}}{% endif %}</div>
               {% endif %}{% if ai_preds %}
               <div class="weather-info">
                  <strong>24-Hour Temperature Forecast</strong>
                  <canvas id="aiChart"></canvas>
               </div>
               {% endif %}{% if wind_preds %}
               <div class="weather-info">
                  <strong>24-Hour Wind Forecast</strong>
                  <canvas id="windChart"></canvas>
               </div>
               {% endif %}
            </div>
         </div>
         {% endif %}
         <div class="hazard-grid">
            {% if city and fire_summary %}
            <div class="risk-card {{fire_risk.lower()}}">
               <div class="card-header">
                  <h3>🔥Fire Hazard</h3>
                  <span class="severity-badge {{fire_summary['severity']}}">{{fire_summary['severity']}}</span>
               </div>
               {% for w in fire_summary['warnings'] %}
               <div class="warning-box">{{w}}</div>
               {% endfor %}
               <div class="summary-section">
                  <h4>Conditions</h4>
                  <div class="context-text">{{fire_summary['context']}}</div>
               </div>
               <div class="summary-section">
                  <h4>Actions</h4>
                  <ul class="action-list">
                     {% for action in fire_summary['user_actions'][:5] %}
                     <li>{{action}}</li>
                     {% endfor %}
                  </ul>
               </div>
            </div>
            <div class="risk-card {{flood_risk.lower()}}">
               <div class="card-header">
                  <h3>🌊Flood Hazard</h3>
                  <span class="severity-badge {{flood_summary['severity']}}">{{flood_summary['severity']}}</span>
               </div>
               {% for w in flood_summary['warnings'] %}
               <div class="warning-box">{{w}}</div>
               {% endfor %}
               <div class="summary-section">
                  <h4>Conditions</h4>
                  <div class="context-text">{{flood_summary['context']}}</div>
               </div>
               <div class="summary-section">
                  <h4>Actions</h4>
                  <ul class="action-list">
                     {% for action in flood_summary['user_actions'][:5] %}
                     <li>{{action}}</li>
                     {% endfor %}
                  </ul>
               </div>
            </div>
            {% if topography_risk %}
            <div class="risk-card {{topography_risk['severity'].lower()}}">
               <div class="card-header">
                  <h3>🏔️ Topography Risk</h3>
                  <span class="severity-badge {{topography_risk['severity']}}">{{topography_risk['severity']}}</span>
               </div>
               <div class="summary-section">
                  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
                     <div class="weather-info"><strong>Landslide:</strong> {{"%.0f"|format(topography_risk['landslide_risk']*100)}}%</div>
                     <div class="weather-info"><strong>Erosion:</strong> {{"%.0f"|format(topography_risk['erosion_risk']*100)}}%</div>
                     <div class="weather-info"><strong>Coastal:</strong> {{"%.0f"|format(topography_risk['coastal_flood_risk']*100)}}%</div>
                     <div class="weather-info"><strong>Drainage:</strong> {{"%.0f"|format(topography_risk['drainage_risk']*100)}}%</div>
                  </div>
               </div>
            </div>
            {% endif %}
            <div class="risk-card low">
               <div class="card-header">
                  <h3>What-If Simulator</h3>
               </div>
               <div class="summary-section">
                  <p style="color:#c8e6c9;margin-bottom:15px">Adjust parameters:</p>
                  <div style="display:grid;grid-template-columns:1fr 1fr;gap:15px">
                     <div><label style="color:#81c784;font-weight:600">Temperature (°F)</label><input type="range" id="sim-temp" min="30" max="110" value="{{forecast['temperature'] if forecast else 70}}"><span id="sim-temp-val" style="color:#a5d6a7">{{forecast['temperature'] if forecast else 70}}°F</span></div>
                     <div><label style="color:#81c784;font-weight:600">Humidity (%)</label><input type="range" id="sim-humidity" min="10" max="100" value="60"><span id="sim-humidity-val" style="color:#a5d6a7">60%</span></div>
                     <div><label style="color:#81c784;font-weight:600">Wind (mph)</label><input type="range" id="sim-wind" min="0" max="50" value="5"><span id="sim-wind-val" style="color:#a5d6a7">5 mph</span></div>
                     <div><label style="color:#81c784;font-weight:600">Precip (mm)</label><input type="range" id="sim-precip" min="0" max="100" value="0"><span id="sim-precip-val" style="color:#a5d6a7">0 mm</span></div>
                  </div>
                  <button onclick="runSimulation()" style="margin-top:15px;width:100%">Run Simulation</button>
                  <div id="simulation-results" style="margin-top:15px;display:none">
                     <h4 style="color:#81c784">Results</h4>
                     <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
                        <div class="weather-info"><strong>Fire:</strong> <span id="fire-change"></span></div>
                        <div class="weather-info"><strong>Flood:</strong> <span id="flood-change"></span></div>
                     </div>
                  </div>
               </div>
            </div>
            {% endif %}
         </div>
         {% if city and lat %}<script>var map=L.map('map').setView([{{lat}},{{lon}}],11);L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap'}).addTo(map);L.marker([{{lat}},{{lon}}]).addTo(map);{% if ai_preds %}new Chart(document.getElementById('aiChart').getContext('2d'),{type:'line',data:{labels:Array.from({length:24},(_,i)=>`${i+1}h`),datasets:[{label:'Temperature (°F)',data:{{ai_preds|tojson}},borderColor:'#66bb6a',backgroundColor:'rgba(102,187,106,0.1)',tension:0.4,fill:true}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#c8e6c9'}}},scales:{y:{ticks:{color:'#c8e6c9'}},x:{ticks:{color:'#c8e6c9'}}}}});{% endif %}{% if wind_preds %}new Chart(document.getElementById('windChart').getContext('2d'),{type:'bar',data:{labels:Array.from({length:24},(_,i)=>`${i+1}h`),datasets:[{label:'Wind (mph)',data:{{wind_preds|tojson}},backgroundColor:'rgba(102,187,106,0.6)'}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#c8e6c9'}}},scales:{y:{ticks:{color:'#c8e6c9'}},x:{ticks:{color:'#c8e6c9'}}}}});{% endif %}document.getElementById('sim-temp').oninput=function(){document.getElementById('sim-temp-val').textContent=this.value+'°F'};document.getElementById('sim-humidity').oninput=function(){document.getElementById('sim-humidity-val').textContent=this.value+'%'};document.getElementById('sim-wind').oninput=function(){document.getElementById('sim-wind-val').textContent=this.value+' mph'};document.getElementById('sim-precip').oninput=function(){document.getElementById('sim-precip-val').textContent=this.value+' mm'};async function runSimulation(){const city="{{city}}";if(!city){alert("Select a city first!");return}const params={city:city,temperature:parseFloat(document.getElementById('sim-temp').value),humidity:parseFloat(document.getElementById('sim-humidity').value),wind_speed:parseFloat(document.getElementById('sim-wind').value),precipitation:parseFloat(document.getElementById('sim-precip').value)};try{const response=await fetch('/whatif',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(params)});const data=await response.json();if(data.error){alert("Error: "+data.error);return}const fireChange=((data.simulated.fire_risk-data.original.fire_risk)*100).toFixed(1);const floodChange=((data.simulated.flood_risk-data.original.flood_risk)*100).toFixed(1);document.getElementById('fire-change').innerHTML=Math.abs(fireChange)<0.1?`<span style="color:#a5d6a7">No Change</span>`:fireChange>0?`<span style="color:#ef5350">↑ +${fireChange}%</span>`:`<span style="color:#66bb6a">↓ ${fireChange}%</span>`;document.getElementById('flood-change').innerHTML=Math.abs(floodChange)<0.1?`<span style="color:#a5d6a7">No Change</span>`:floodChange>0?`<span style="color:#ef5350">↑ +${floodChange}%</span>`:`<span style="color:#66bb6a">↓ ${floodChange}%</span>`;document.getElementById('simulation-results').style.display='block'}catch(error){alert("Simulation failed: "+error)}}</script>{% endif %}
      </div>
   </body>
</html>
""", cities=sorted(city_coords.keys()), city=city, lat=lat, lon=lon, forecast=forecast, ai_preds=ai_preds, wind_preds=wind_preds, fire_risk=fire_risk, flood_risk=flood_risk, fire_summary=fire_summary, flood_summary=flood_summary, activities=activities, topography_risk=topography_risk, error_message=error_message)
if __name__ == "__main__":
    print("="*60)
    print("Starting EnviroHazardLI")
    print("Visit: http://localhost:5000")
    print("="*60)
    app.run(debug=True, host="0.0.0.0", port=5000)