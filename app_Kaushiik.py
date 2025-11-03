import streamlit as st
import pandas as pd
import requests
import nbformat
from nbconvert import PythonExporter
from datetime import datetime

st.set_page_config(layout="wide", page_title="TripGuard", page_icon="🛡️")

st.markdown("""
    <style>
        header {visibility: hidden;}
        .block-container {padding-top: 2rem;}
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(to bottom right, #e6f3ff, #ffffff);
        }
        [data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0);
        }
        .stApp {
            color: #00264d;
        }
        .card, .light-card {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-top: 15px;
        }
        .subhead {
            font-size: 1.3em;
            margin-bottom: 10px;
            color: #004080;
        }
        .forecast-title {
            font-size: 1.2em;
            font-weight: 600;
            color: #0066cc;
            text-align: center;
        }
        h1, h2, h3, h4, h5, h6, label, span, p {
            color: #00264d !important;
        }
        div[data-testid="stDataFrame"] {
            background-color: #ffffff !important;
            border-radius: 10px;
            color: #001f3f !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        }
        table {
            background-color: #ffffff !important;
            color: #001f3f !important;
        }
        .score-value {
            text-align: center;
            font-size: 42px;
            font-weight: 700;
            color: #0047ab;
            background-color: #e6f0ff;
            border-radius: 10px;
            padding: 8px 0;
            margin-top: 5px;
        }
    </style>
""", unsafe_allow_html=True)


st.title("🛡️ TripGuard — AI Travel Safety Predictor")

# ---------- INPUT SECTION ----------
col1, col2 = st.columns(2)
with col1:
    city = st.text_input("Enter City")
with col2:
    state = st.text_input("Enter State / Region")

# ---------- WEATHER FORECAST ----------
if city and state:
    st.markdown(f"<div class='subhead'>5-Day Forecast for {city}, {state}</div>", unsafe_allow_html=True)

    API_KEY = "c41cab54a6553b6d92b8e1904177fb15"  # Replace with your OpenWeather API key
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city},{state},IN&appid={API_KEY}&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        forecasts = []
        for item in data["list"]:
            dt = datetime.fromtimestamp(item["dt"])
            temp = item["main"]["temp"]
            weather = item["weather"][0]["description"].title()
            forecasts.append({"Date": dt.date(), "Time": dt.strftime("%H:%M"), "Temp (°C)": temp, "Weather": weather})

        df = pd.DataFrame(forecasts)
        days = sorted(df["Date"].unique())[:5]
        cols = st.columns(5)

        for i, day in enumerate(days):
            day_df = df[df["Date"] == day][["Time", "Temp (°C)", "Weather"]].reset_index(drop=True)
            with cols[i]:
                st.markdown(f"<div class='forecast-title'>{day.strftime('%A')}</div>", unsafe_allow_html=True)
                st.dataframe(day_df, use_container_width=True, hide_index=True)
    else:
        st.error("City not found or API error.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- SAFETY SCORE PREDICTION ----------

def load_notebook_model():
    with open("ML_Project.ipynb") as f:
        nb = nbformat.read(f, as_version=4)
    exporter = PythonExporter()
    source, _ = exporter.from_notebook_node(nb)
    code = compile(source, "ML_Project.ipynb", 'exec')
    scope = {}
    exec(code, scope)
    return scope.get("predict_safety")

@st.cache_resource
def get_predict_function():
    try:
        return load_notebook_model()
    except Exception as e:
        st.error(f"Failed to load ML model: {e}")
        return None

predict_safety = get_predict_function()

if city and state:
    try:
        result = predict_safety(state)
        if "error" in result:
            st.warning(result["error"])
        else:
            st.markdown(f"<div class='subhead'>Safety Prediction for {state}</div>", unsafe_allow_html=True)

            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Features Used**")
                features_df = pd.DataFrame({"Feature": result["features_used"]})
                st.dataframe(features_df, use_container_width=True, hide_index=True)
            
            with cols[1]:
                st.markdown("<div style='text-align:center; font-weight:bold; color:#ffcc00;'>Actual Score</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align:center; font-size:40px; font-weight:bold; color:#33ff77;'>{result['actual_score']}</div>", unsafe_allow_html=True)
                st.markdown("<div style='text-align:center; font-weight:bold; color:#ffcc00;'>Predicted Score</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align:center; font-size:40px; font-weight:bold; color:#33ff77;'>{result['predicted_score']}</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Model error: {e}")

