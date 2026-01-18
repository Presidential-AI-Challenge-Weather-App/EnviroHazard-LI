#  EnviroHazardLI  
### AI Assisted Environmental Hazard Monitoring for Long Island

EnviroHazardLI is a real time environmental hazard assessment tool designed specifically for Long Island. It combines live weather data, elevation analysis, and simple machine-learning or rule-based calculations to provide clear fire and flood hazard insights for residents, local governments, and outdoor activity planning.

---

##  Features

### **Real-Time Hazard Monitoring**

####  Fire Risk Assessment  
Based on temperature, humidity, wind speed, recent precipitation, drought conditions, and vegetation dryness.

####  Flood Risk Assessment  
Uses precipitation amounts, soil saturation logic, elevation, drainage capacity, and urbanization factors.

---

##  AI-Assisted Predictions
Uses a neural network if a trained model is available  
If not, the system falls back to an easier engine

---

##  Integrated Data Sources

### **NOAA Weather API
Live forecasts  
Hourly temperature, humidity, wind, dew point, and precipitation indicators  

### **USGS Elevation Services**
Elevation lookups for flood susceptibility  
Cached since terrain does not change  

### **Derived Environmental Metrics**
Estimated soil moisture  
Runoff index  
Drought score 

---

##  Hazard Recommendations
**Resident Guidance** – Steps to stay safe  
**Government Recommendations** – Proactive community-level actions  
**Outdoor Activity Ratings** – Activities scored as Safe / Caution / Avoid  

---

## Visualizations
Leaflet.js interactive dark map
24-hour temperature line chart  
24-hour wind-speed bar chart  
Color-coded hazards  

---

#### **Fire Summaries**
Analyzes:
Temperature extremes and heat stress
Humidity levels and fuel dryness
Wind speeds and fire spread potential
Drought duration and vegetation

#### **Flood Summaries**
Analyzes:
Rainfall amounts and intensity
Soil saturation and absorption capacity
Topography and elevation vulnerabilities
Stream and waterway capacity

#### **Activity Recommendations**
Evaluates outdoor activities based on:
Temperature comfort ranges
Wind safety thresholds
Fire risk levels
Flood risk levels
Humidity conditions
Precipitation amounts

Each activity is categorized as:
**Safe**
**Caution**
**Not Recommended**

---

## Supported Locations

Currently supports 12 Long Island towns with precise coordinates:

| Town | Latitude | Longitude |
|------|----------|-----------|
| Commack | 40.8429 | -73.2920 |
| Smithtown | 40.8559 | -73.2007 |
| Huntington | 40.8682 | -73.4257 |
| Babylon | 40.7009 | -73.3257 |
| Islip | 40.7301 | -73.2104 |
| Southampton | 40.8843 | -72.3895 |
| Riverhead | 40.9170 | -72.6620 |
| Patchogue | 40.7651 | -73.0151 |
| Bay Shore | 40.7251 | -73.2451 |
| Centereach | 40.8565 | -73.0818 |
| Hicksville | 40.7682 | -73.5251 |
| Hempstead | 40.7062 | -73.6187 |

---
**Built for Long Island communities**

--- 
**Maintained By:** EnviroHazardLI Team - Lucas Kuriakose and Michael Wagner
