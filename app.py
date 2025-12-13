import os
import re
import time
import requests
from flask import Flask, render_template, request

app = Flask(__name__, template_folder="templates")

# Long Island towns
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

def get_weather(lat, lon):
    """Get weather from NOAA"""
    try:
        headers = {"User-Agent": "(EnviroHazardLI, contact@example.com)"}
        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        r = requests.get(points_url, headers=headers, timeout=10)
        r.raise_for_status()
        
        forecast_url = r.json().get("properties", {}).get("forecast")
        if not forecast_url:
            return None
            
        time.sleep(0.5)
        r2 = requests.get(forecast_url, headers=headers, timeout=10)
        r2.raise_for_status()
        
        periods = r2.json().get("properties", {}).get("periods", [])
        if periods:
            return periods[0]
    except:
        pass
    return None

def calculate_risks(forecast):
    """Simple risk calculation based on weather"""
    if not forecast:
        return {
            "fire_risk": "LOW",
            "flood_risk": "LOW",
            "fire_score": 0.2,
            "flood_score": 0.2
        }
    
    temp = forecast.get("temperature", 70)
    wind_str = forecast.get("windSpeed", "5 mph")
    wind = int(re.search(r'(\d+)', wind_str).group(1)) if re.search(r'(\d+)', wind_str) else 5
    
    short_forecast = forecast.get("shortForecast", "").lower()
    detailed = forecast.get("detailedForecast", "").lower()
    
    # Fire risk
    fire_score = 0.0
    if temp > 85:
        fire_score += 0.3
    if temp > 95:
        fire_score += 0.2
    if wind > 15:
        fire_score += 0.2
    if wind > 25:
        fire_score += 0.2
    if "dry" in short_forecast or "dry" in detailed:
        fire_score += 0.1
        
    # Flood risk
    flood_score = 0.0
    rain_words = ["rain", "storm", "shower", "flood", "heavy"]
    for word in rain_words:
        if word in short_forecast or word in detailed:
            flood_score += 0.2
    if "heavy" in short_forecast or "heavy" in detailed:
        flood_score += 0.3
    if "thunderstorm" in short_forecast or "thunderstorm" in detailed:
        flood_score += 0.2
        
    fire_score = min(fire_score, 1.0)
    flood_score = min(flood_score, 1.0)
    
    return {
        "fire_risk": "HIGH" if fire_score > 0.6 else "MODERATE" if fire_score > 0.3 else "LOW",
        "flood_risk": "HIGH" if flood_score > 0.6 else "MODERATE" if flood_score > 0.3 else "LOW",
        "fire_score": fire_score,
        "flood_score": flood_score
    }

def get_fire_summary(fire_score, temp, wind, forecast):
    """Detailed fire risk analysis"""
    summary = {
        "severity": "LOW",
        "warnings": [],
        "context": "",
        "user_actions": [],
        "government_actions": []
    }
    
    if fire_score > 0.7:
        summary["severity"] = "HIGH"
        summary["warnings"].append("High fire risk - Dangerous for outdoor burning")
    elif fire_score > 0.4:
        summary["severity"] = "MODERATE"
        summary["warnings"].append("Moderate fire risk - Exercise caution")
    else:
        summary["severity"] = "LOW"
        summary["warnings"].append("Low fire risk - Conditions look favorable")
    
    # Context
    parts = ["Fire Risk Analysis:"]
    if temp > 95:
        parts.append(f"Very hot ({temp}°F) raises ignition risk.")
    elif temp > 85:
        parts.append(f"Warm ({temp}°F) and drying conditions.")
    else:
        parts.append(f"Temperature ({temp}°F) is comfortable.")
    
    if wind > 25:
        parts.append(f"High winds ({wind} mph) could spread fires rapidly.")
    elif wind > 15:
        parts.append(f"Moderate winds ({wind} mph) may accelerate spread.")
    else:
        parts.append(f"Light winds ({wind} mph) limit fire spread.")
    
    summary["context"] = " ".join(parts)
    
    # Actions based on severity
    if fire_score > 0.6:
        summary["user_actions"] = [
            "No outdoor burning - cancel campfires and burn permits",
            "Clear dry brush 30 feet from structures",
            "Water vulnerable landscaping",
            "Sign up for local emergency alerts",
            "Prepare an evacuation kit"
        ]
        summary["government_actions"] = [
            "Issue burn bans and public alerts",
            "Stage firefighting resources in high-risk zones",
            "Increase patrols and inspections",
            "Deploy public safety messaging"
        ]
    elif fire_score > 0.3:
        summary["user_actions"] = [
            "Delay non-essential outdoor fires",
            "Clean gutters and remove dry debris",
            "Supervise grills carefully",
            "Water lawns and plants regularly"
        ]
        summary["government_actions"] = [
            "Increase monitoring and public education",
            "Prepare resources for rapid deployment",
            "Issue guidance on defensible space"
        ]
    else:
        summary["user_actions"] = [
            "Maintain safe practices with open flames",
            "Regularly remove dry leaf litter",
            "Stay informed during dry spells"
        ]
        summary["government_actions"] = [
            "Continue routine inspections",
            "Promote fire safety programs"
        ]
    
    return summary

def get_flood_summary(flood_score, forecast):
    """Detailed flood risk analysis"""
    summary = {
        "severity": "LOW",
        "warnings": [],
        "context": "",
        "user_actions": [],
        "government_actions": []
    }
    
    if flood_score > 0.7:
        summary["severity"] = "HIGH"
        summary["warnings"].append("High flood risk - Significant flooding expected")
    elif flood_score > 0.4:
        summary["severity"] = "MODERATE"
        summary["warnings"].append("Moderate flood risk - Localized flooding possible")
    else:
        summary["severity"] = "LOW"
        summary["warnings"].append("Low flood risk - Normal drainage")
    
    # Context
    short = forecast.get("shortForecast", "").lower() if forecast else ""
    detailed = forecast.get("detailedForecast", "").lower() if forecast else ""
    
    parts = ["Flood Risk Analysis:"]
    if "heavy" in short or "heavy" in detailed:
        parts.append("Heavy rainfall may overwhelm drainage systems.")
    elif "rain" in short or "rain" in detailed:
        parts.append("Rainfall creating elevated runoff.")
    else:
        parts.append("Little rain expected - low flood risk.")
    
    if "thunderstorm" in short or "thunderstorm" in detailed:
        parts.append("Thunderstorms can produce rapid flooding.")
    
    summary["context"] = " ".join(parts)
    
    # Actions based on severity
    if flood_score > 0.6:
        summary["user_actions"] = [
            "Do not drive through flooded roads - Turn Around, Don't Drown",
            "Move to higher ground if in low areas",
            "Elevate electronics and valuables",
            "Monitor local emergency alerts",
            "Keep devices charged for emergencies"
        ]
        summary["government_actions"] = [
            "Activate emergency operations and rescue teams",
            "Close flooded roads and deploy signage",
            "Stage swift-water rescue resources",
            "Issue emergency notifications"
        ]
    elif flood_score > 0.3:
        summary["user_actions"] = [
            "Avoid flood-prone spots and underpasses",
            "Clear gutters and downspouts",
            "Move vehicles to higher ground if needed",
            "Clear yard drains",
            "Test sump pumps"
        ]
        summary["government_actions"] = [
            "Inspect and clean storm drains",
            "Pre-position pumps and barriers",
            "Monitor rainfall and stream levels",
            "Issue flood watches"
        ]
    else:
        summary["user_actions"] = [
            "Keep gutters clear",
            "Know your local flood risk zone",
            "Maintain yard drainage"
        ]
        summary["government_actions"] = [
            "Routine storm drain maintenance",
            "Promote flood preparedness education"
        ]
    
    return summary

def get_warnings(risks, forecast):
    """Generate warnings based on risk levels"""
    fire_warnings = []
    flood_warnings = []
    
    if risks["fire_risk"] == "HIGH":
        fire_warnings.append("High fire danger - no outdoor burning")
        fire_warnings.append("Clear dry brush from around buildings")
    elif risks["fire_risk"] == "MODERATE":
        fire_warnings.append("Moderate fire risk - be careful with flames")
    else:
        fire_warnings.append("Low fire risk - normal precautions")
    
    if risks["flood_risk"] == "HIGH":
        flood_warnings.append("High flood risk - avoid low-lying areas")
        flood_warnings.append("Do not drive through flooded roads")
    elif risks["flood_risk"] == "MODERATE":
        flood_warnings.append("Moderate flood risk - monitor conditions")
    else:
        flood_warnings.append("Low flood risk - normal conditions")
    
    return fire_warnings, flood_warnings

def get_activities(risks, forecast):
    """Recommend activities based on conditions"""
    temp = forecast.get("temperature", 70) if forecast else 70
    
    safe = []
    caution = []
    not_recommended = []
    
    # Simple activity logic
    if risks["fire_risk"] == "LOW" and risks["flood_risk"] == "LOW" and 60 < temp < 85:
        safe = ["Hiking", "Beach", "Camping", "Cycling", "Grilling", "Fishing", "Picnicking"]
    elif risks["fire_risk"] == "HIGH" or risks["flood_risk"] == "HIGH":
        not_recommended = ["Camping", "Hiking in woods", "Outdoor grilling"]
        caution = ["Beach activities", "Short walks"]
        safe = ["Indoor activities"]
    else:
        caution = ["Hiking", "Camping", "Grilling"]
        safe = ["Beach", "Cycling", "Fishing"]
    
    return {"safe": safe, "caution": caution, "not_recommended": not_recommended}

@app.route("/", methods=["GET", "POST"])
def index():
    town = None
    lat = lon = None
    forecast = None
    fire_risk = "LOW"
    flood_risk = "LOW"
    fire_warnings = []
    flood_warnings = []
    fire_summary = None
    flood_summary = None
    activities = None
    error_message = None
    
    if request.method == "POST":
        town = request.form.get("city")
        if town and town in TOWNS:
            try:
                lat, lon = TOWNS[town]
                forecast = get_weather(lat, lon)
                risks = calculate_risks(forecast)
                
                fire_risk = risks["fire_risk"]
                flood_risk = risks["flood_risk"]
                fire_warnings, flood_warnings = get_warnings(risks, forecast)
                
                # Get detailed summaries
                temp = forecast.get("temperature", 70) if forecast else 70
                wind_str = forecast.get("windSpeed", "5 mph") if forecast else "5 mph"
                wind = int(re.search(r'(\d+)', wind_str).group(1)) if re.search(r'(\d+)', wind_str) else 5
                
                fire_summary = get_fire_summary(risks["fire_score"], temp, wind, forecast)
                flood_summary = get_flood_summary(risks["flood_score"], forecast)
                activities = get_activities(risks, forecast)
            except Exception as e:
                error_message = f"Error getting weather data: {str(e)}"
        else:
            error_message = "Please select a valid town"
    
    return render_template("index.html",
        cities=sorted(TOWNS.keys()),
        city=town,
        lat=lat,
        lon=lon,
        forecast=forecast,
        fire_risk=fire_risk,
        fire_warnings=fire_warnings,
        flood_risk=flood_risk,
        flood_warnings=flood_warnings,
        fire_summary=fire_summary,
        flood_summary=flood_summary,
        activities=activities,
        error_message=error_message)

app_handler = app

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
