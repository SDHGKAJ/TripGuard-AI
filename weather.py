import requests
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("tripguard")

# OpenWeatherMap API configuration
WEATHER_API_KEY = "c41cab54a6553b6d92b8e1904177fb15"
BASE_URL = "http://api.openweathermap.org/data/2.5/forecast"

# Indian state capitals mapping
STATE_CITIES = {
    "ANDHRA PRADESH": "Amaravati",
    "KARNATAKA": "Bengaluru",
    "KERALA": "Thiruvananthapuram",
    "TAMIL NADU": "Chennai",
    "TELANGANA": "Hyderabad",
    "MAHARASHTRA": "Mumbai",
    "DELHI": "Delhi",
    "GUJARAT": "Ahmedabad",
    "WEST BENGAL": "Kolkata",
    "PUNJAB": "Chandigarh",
    "UTTAR PRADESH": "Lucknow",
    "BIHAR": "Patna",
    "MADHYA PRADESH": "Bhopal",
    "RAJASTHAN": "Jaipur"
}

def get_5day_forecast(state: str) -> list:
    """Get 5-day weather forecast for a given Indian state using its capital city."""
    if not WEATHER_API_KEY:
        return [{"error": "Weather API key not configured"}]
    
    state = state.upper().strip()
    if state not in STATE_CITIES:
        return [{"error": f"City not found for state: {state}"}]
    
    city = STATE_CITIES[state]
    params = {
        "q": city,
        "appid": WEATHER_API_KEY,
        "units": "metric",  # Use Celsius
        "cnt": 40  # Get full 5 days (8 measurements per day)
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Process and simplify the forecast data
        forecasts = []
        seen_dates = set()
        
        for item in data.get("list", []):
            dt = datetime.fromtimestamp(item["dt"])
            date_str = dt.strftime("%Y-%m-%d")
            
            # Only take one reading per day
            if date_str in seen_dates:
                continue
            seen_dates.add(date_str)
            
            forecast = {
                "date": date_str,
                "temp_max": round(item["main"]["temp_max"], 1),
                "temp_min": round(item["main"]["temp_min"], 1),
                "humidity": item["main"]["humidity"],
                "description": item["weather"][0]["description"].capitalize(),
                "wind_speed": round(item["wind"]["speed"] * 3.6, 1)  # Convert m/s to km/h
            }
            forecasts.append(forecast)
            
            # Stop after 5 days
            if len(forecasts) >= 5:
                break
                
        return forecasts
        
    except requests.RequestException as e:
        logger.error(f"Weather API error: {e}")
        return [{"error": f"Failed to fetch weather data: {str(e)}"}]