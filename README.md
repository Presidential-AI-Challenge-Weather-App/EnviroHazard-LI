#  EnviroHazardLI  
### AI-Assisted Environmental Hazard Monitoring for Long Island

EnviroHazardLI is a real-time environmental hazard assessment tool designed specifically for Long Island. It combines live weather data, elevation analysis, and simple machine-learning or rule-based calculations to provide clear fire and flood hazard insights for residents, local governments, and outdoor activity planning.

---

##  Features

### **Real-Time Hazard Monitoring**

####  Fire Risk Assessment  
Based on temperature, humidity, wind speed, recent precipitation, drought conditions, and vegetation dryness.

####  Flood Risk Assessment  
Uses precipitation amounts, soil saturation logic, elevation, drainage capacity, and urbanization factors.

---

##  AI-Assisted Predictions
Uses a **lightweight neural network** if a trained model is available  
If not, the system falls back to a **robust rule-based engine**

---

##  Integrated Data Sources

### **NOAA Weather API**
Live forecasts  
Hourly temperature, humidity, wind, dew point, and precipitation indicators  

### **USGS Elevation Services**
Elevation lookups for flood susceptibility  
Cached since terrain does not change  

### **Derived Environmental Metrics**
Estimated soil moisture  
Runoff index  
Drought score  
Simplified urbanization factors  

---

##  Actionable Hazard Recommendations
**Resident Guidance** – Steps to stay safe  
**Government Recommendations** – Proactive community-level actions  
**Outdoor Activity Ratings** – 12 activities scored as Safe / Caution / Avoid  

---

## 🗺 Interactive Visualizations
Leaflet.js interactive dark map  
24-hour temperature line chart  
24-hour wind-speed bar chart  
Color-coded hazard severity badges  

---

#  Quick Start

## **Prerequisites**
Python 3.8+  
pip  

## **Installation**
bash
git clone https://github.com/Congressional-App-Challenge-Weather-App/EnviroHazardLI/
pip install flask requests numpy pandas scikit-learn tensorflow

3. **Run the application**
bash
   python app.py


4. **Open your browser**
   http://localhost:5000


---

##  Project StructureEnviroHazardLI/
├── models/                           # AI/ML Models Directory
│   ├── fire_risk_regressor.h5       # Fire risk prediction model
│   ├── flood_risk_regressor.h5      # Flood risk prediction model
│   ├── hazard_classifier.h5         # Hazard classification model
│   ├── hazard_multioutput.h5        # Multi-output combined model
│   └── scaler_and_encoders.pkl      # Preprocessing artifacts (StandardScaler, OneHotEncoder, LabelEncoder)
│
├── templates/                        # HTML Templates
│   └── index.html                   # Main dashboard template
│
├── app.py                           # Main Flask application
├── NeuralNetwork.py                 # Neural network training script
├── Generate_CSV.py                  # Training data generation script
├── hazard_dummy_data_v2.csv        # Sample training dataset
└── requirements.txt                 # Python dependencies
---

##  Usage Guide

### **Basic Operation**
1. Launch the application: `python app.py`
2. Navigate to `http://localhost:5000` in your web browser
3. Select a Long Island town from the dropdown menu
4. Click "Analyze Hazards" to view comprehensive risk assessment

### **Understanding the Dashboard**

#### **Left Side: Interactive Map**
- Dark-themed Leaflet map centered on selected town
- Marker indicating the exact location
- 60% of screen width for optimal viewing

#### **Right Side: Weather Sidebar**
- **Current Conditions** - Real-time weather from NOAA
- **Coordinates** - Latitude and longitude
- **24-Hour Temperature Forecast** - Line chart with actual NOAA hourly data
- **24-Hour Wind Forecast** - Bar chart showing wind speed predictions
- **Activity Recommendations** - Color-coded list of outdoor activities:
  - ✓ Green = Safe & Recommended
  - ⚠ Orange = Use Caution
  - ✗ Red = Not Recommended

#### **Bottom: Hazard Assessment Cards**

**Fire Hazard Assessment Card:**
- **Severity Badge** - LOW/MODERATE/HIGH/EXTREME with color coding
- **Warning Messages** - Critical alerts based on conditions
- **Current Conditions Analysis** - Detailed explanation of fire risk factors:
  - Temperature impact on fire danger
  - Humidity levels and fuel flammability
  - Wind speed and ember spread potential
  - Drought conditions and vegetation status
- **Resident Actions** - Top 5 immediate safety measures
- **Government Actions** - Top 5 emergency management steps

**Flood Hazard Assessment Card:**
- **Severity Badge** - LOW/MODERATE/HIGH/EXTREME with color coding
- **Warning Messages** - Critical flooding alerts
- **Current Conditions Analysis** - Detailed explanation of flood risk factors:
  - Precipitation amounts and intensity
  - Soil saturation levels
  - Elevation and topography
  - Urban development and drainage capacity
  - Stream and waterway status
- **Resident Actions** - Top 5 flood safety measures
- **Government Actions** - Top 5 flood response strategies

---

## How It Works

### **Data Collection Pipeline**

1. **NOAA Weather API Integration**
   - Fetches point forecast for selected coordinates
   - Retrieves 24 hours of hourly weather data
   - Extracts temperature, humidity, wind speed, dewpoint
   - Analyzes forecast text for precipitation keywords

2. **USGS Elevation Service**
   - Queries elevation data for terrain analysis
   - Used for flood susceptibility calculations
   - Cached permanently (elevation doesn't change)

3. **Feature Engineering**
   - **Meteorological Features**: temperature, humidity, wind_speed, precipitation
   - **Derived Features**: days_since_last_rain, rainy_hours
   - **Topographical Features**: elevation, slope, aspect, distance_to_water
   - **Urban Features**: population_density, urbanization_index, impervious_surface_ratio
   - **Environmental Indices**: 
     - soil_moisture (based on precipitation, humidity, dry days)
     - surface_runoff (precipitation × slope factor)
     - streamflow_index (precipitation + elevation + soil moisture)
     - drought_index (temperature + humidity + dry days)

### **Risk Calculation Methods**

#### **AI Model Mode** (if models exist in /models/ directory)
- **Neural Network Architecture**:
  - Input: 24 engineered features
  - Hidden layers with dropout for regularization
  - Multi-output architecture:
    - Output 1: Fire risk probability (0-1)
    - Output 2: Flood risk probability (0-1)
    - Output 3: Hazard classification (low/moderate/high)
- **Preprocessing**:
  - StandardScaler for numerical features
  - OneHotEncoder for categorical features (soil_type)
  - LabelEncoder for hazard class

#### **Rule-Based Mode** (fallback if no model)
- **Fire Risk Calculation**:
  fire_risk = 0
  if temp > 90: fire_risk += 0.25
  if temp > 100: fire_risk += 0.25
  fire_risk += (60 - humidity) / 60 * 0.3
  fire_risk += (wind_speed / 30) * 0.2


- **Flood Risk Calculation**:
  flood_risk = precipitation_24h / 40 * 0.6
  flood_risk += (1 - soil_moisture) * 0.2


### **Recommendation Generation**

#### **Fire Summaries**
Analyzes:
Temperature extremes and heat stress
Humidity levels and fuel dryness
Wind speeds and fire spread potential
Drought duration and vegetation fuel loads

Generates context-aware messages and severity-scaled actions:
**EXTREME** (>85%): Emergency protocols, evacuations, burn bans
**HIGH** (70-85%): Preventive measures, increased patrols
**MODERATE** (40-70%): Caution advisories, maintenance
**LOW** (<40%): Standard fire safety practices

#### **Flood Summaries**
Analyzes:
Rainfall amounts and intensity
Soil saturation and absorption capacity
Topography and elevation vulnerabilities
Urban infrastructure and drainage systems
Stream and waterway capacity

Generates context-aware messages and severity-scaled actions:
**EXTREME** (>85%): Evacuations, road closures, emergency operations
**HIGH** (70-85%): Flood barriers, rescue team staging
**MODERATE** (40-70%): Monitoring, drainage clearing
**LOW** (<40%): Routine maintenance, preparedness

#### **Activity Recommendations**
Evaluates 12 outdoor activities based on:
Temperature comfort ranges
Wind safety thresholds
Fire risk levels
Flood risk levels
Humidity conditions
Precipitation amounts

Each activity categorized as:
**Safe** - All conditions favorable
**Caution** - Some risk factors present, specific warnings given
**Not Recommended** - Unsafe conditions detected

---

## 🗺️ Supported Locations

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



##  Technical Specifications

### **Frontend**
HTML5, CSS3, JavaScript
Leaflet 1.9.x
Chart.js 4.x
Responsive design (desktop-optimized)
No build process required

### **Backend**
Python 3.8+
Flask 2.0+
RESTful architecture
Template rendering
In-memory caching

### **Machine Learning** (optional)
TensorFlow 2.x
Keras Sequential API
Multi-output neural network
StandardScaler preprocessing
OneHotEncoder for categoricals

### **Data Science Stack**
NumPy 1.21+
Pandas 1.3+
scikit-learn 1.0+

### **External APIs**
NOAA Weather API v1
USGS Elevation Service v1

---

##  Performance

### **Optimization Techniques**
API response caching (10 min)
Elevation data caching (permanent)
Minimal external dependencies
Efficient feature engineering
Lazy loading of ML models

### **Benchmarks** (approximate)
Initial page load: 500ms
City analysis (cached): 200ms
City analysis (fresh): 2-3 seconds
Memory usage: 50MB (without model), 200MB (with model)

### **Scalability**
Single-threaded (Flask development server)
Can handle ~10-20 concurrent users
For production: Use gunicorn with workers
Stateless design enables horizontal scaling

---
**Built for Long Island communities**

Stay safe, stay informed, protect our environment. 

--- 
**Maintained By:** EnviroHazardLI Team
