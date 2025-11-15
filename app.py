# app.py -- full script with corrected SMS alert system
import streamlit as st
import pandas as pd
import numpy as np
import requests
import pydeck as pdk
import plotly.express as px
from datetime import datetime, timedelta
from krigging import perform_kriging_correct, get_aqi_at_location
import geopandas as gpd
from shapely.geometry import Point
import os
import re
import time
import json
import traceback

# -------------------------
# CONFIG / CREDENTIALS
# -------------------------
# Prefer st.secrets, then environment variables, then fallbacks (not recommended)
SMS77_API_KEY = st.secrets.get("SMS77_API_KEY", None) or os.environ.get("SMS77_API_KEY", None) or "ce9196b9famsh41c38d8b9917c08p11f8e0jsnd367c1038fa7"
TWILIO_ACCOUNT_SID = st.secrets.get("TWILIO_ACCOUNT_SID", None) or os.environ.get("TWILIO_ACCOUNT_SID", None) or ""
TWILIO_AUTH_TOKEN = st.secrets.get("TWILIO_AUTH_TOKEN", None) or os.environ.get("TWILIO_AUTH_TOKEN", None) or ""
TWILIO_PHONE_NUMBER = st.secrets.get("TWILIO_PHONE_NUMBER", None) or os.environ.get("TWILIO_PHONE_NUMBER", None) or ""

# SMS throttling window (seconds) - prevents spamming the same number repeatedly
SMS_THROTTLE_SECONDS = 300  # 5 minutes per number by default

# -------------------------
# UTILITIES: Phone validation, send_sms wrapper (SMS77 + optional Twilio fallback)
# -------------------------
PHONE_REGEX = re.compile(r"^\+\d{7,15}$")  # simple E.164-like check (must start with + and digits, length 7-15 digits)

def validate_phone_number(phone: str) -> bool:
    """Simple validation - phone must be in +<countrycode><number> format (digits only after +)."""
    if not isinstance(phone, str):
        return False
    phone = phone.strip()
    return bool(PHONE_REGEX.match(phone))

def _sms77_send_once(phone: str, message: str, api_key: str, timeout=10):
    """
    Attempt to send SMS via SMS77. Returns (success_bool, response_text).
    Uses the SMS77 HTTP API (gateway.sms77.io).
    """
    url = "https://gateway.sms77.io/api/sms"
    payload = {
        "to": phone,
        "text": message,
        "from": "AQIAlert"
    }
    headers = {
        "X-Api-Key": api_key
    }
    try:
        resp = requests.post(url, data=payload, headers=headers, timeout=timeout)
        # SMS77 returns 200 even on some failures; attempt to parse JSON
        text = resp.text
        try:
            j = resp.json()
        except Exception:
            j = None
        if resp.status_code == 200:
            # The service may return JSON with 'success' or similar; fall back to status code
            if j:
                # Standard sms77 returns {"success":1,"errors":[] } or similar
                if j.get("success") in (1, True) or j.get("status") == "success":
                    return True, json.dumps(j)
                # If there are errors included, forward them
                if "errors" in j:
                    return False, json.dumps(j.get("errors"))
            # if JSON not helpful, accept 200 as success with response text
            return True, text
        else:
            return False, f"Status code {resp.status_code} | {text}"
    except Exception as e:
        return False, f"Exception: {str(e)}"

def _twilio_send_once(phone: str, message: str, account_sid: str, auth_token: str, from_number: str, timeout=10):
    """
    Attempt to send via Twilio REST API (HTTP) if credentials are present.
    Returns (success_bool, response_text)
    """
    if not account_sid or not auth_token or not from_number:
        return False, "Twilio credentials not provided"
    try:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
        payload = {
            "To": phone,
            "From": from_number,
            "Body": message
        }
        resp = requests.post(url, data=payload, auth=(account_sid, auth_token), timeout=timeout)
        if resp.status_code in (200, 201):
            return True, resp.text
        else:
            return False, f"Twilio status {resp.status_code} | {resp.text}"
    except Exception as e:
        return False, f"Exception: {str(e)}"

def send_sms(phone: str, message: str, max_retries=2):
    """
    High-level send SMS wrapper:
      - Validates number
      - Throttles repeated sends per phone (session-state)
      - Attempts SMS77 (with retries)
      - Falls back to Twilio if SMS77 fails and Twilio creds present
      - Returns (success_bool, provider_name, response_text)
    """
    phone = phone.strip()
    # Validate
    if not validate_phone_number(phone):
        return False, "validation", "Invalid phone number format. Use +<countrycode><number> (e.g. +919876543210)."

    # Throttle check (session-state)
    now_ts = time.time()
    if "sms_last_sent" not in st.session_state:
        st.session_state["sms_last_sent"] = {}
    last_sent = st.session_state["sms_last_sent"].get(phone, 0)
    if now_ts - last_sent < SMS_THROTTLE_SECONDS:
        wait_secs = int(SMS_THROTTLE_SECONDS - (now_ts - last_sent))
        return False, "throttle", f"Throttled: Please wait {wait_secs} seconds before requesting another SMS to this number."

    # Try SMS77
    if SMS77_API_KEY:
        attempt = 0
        while attempt <= max_retries:
            success, resp = _sms77_send_once(phone, message, SMS77_API_KEY)
            if success:
                st.session_state["sms_last_sent"][phone] = now_ts
                return True, "sms77", resp
            # retry on transient network errors (simple heuristic)
            attempt += 1
            time.sleep(1)
        # SMS77 ultimately failed
        sms77_error = resp
    else:
        sms77_error = "SMS77 API key not configured."

    # Fallback to Twilio if configured
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER:
        attempt = 0
        while attempt <= max_retries:
            success, resp = _twilio_send_once(phone, message, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER)
            if success:
                st.session_state["sms_last_sent"][phone] = now_ts
                return True, "twilio", resp
            attempt += 1
            time.sleep(1)
        # Twilio failed as well
        twilio_err = resp
        return False, "both_failed", f"SMS77 error: {sms77_error} | Twilio error: {twilio_err}"
    else:
        # Twilio not configured, return SMS77 failure reason
        return False, "sms77_failed", f"SMS77 error: {sms77_error}"

# -------------------------
# YOUR EXISTING APP CODE (with minor fixes)
# -------------------------
st.set_page_config(layout="wide", page_title="Delhi Air Quality Dashboard", page_icon="💨")

API_TOKEN = "97a0e712f47007556b57ab4b14843e72b416c0f9"
DELHI_BOUNDS = "28.404,76.840,28.883,77.349"
DELHI_LAT = 28.6139
DELHI_LON = 77.2090

# ---------- Delhi polygon loader (unchanged except guard)
def load_delhi_boundary_from_url():
    url = "https://raw.githubusercontent.com/shuklaneerajdev/IndiaStateTopojsonFiles/master/Delhi.geojson"
    try:
        gdf = gpd.read_file(url)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
        polygon = gdf.unary_union
        return gdf, polygon
    except Exception as e:
        st.error(f"Failed to load Delhi polygon: {e}")
        return None, None

if "delhi_gdf" not in st.session_state or "delhi_polygon" not in st.session_state:
    gdf, polygon = load_delhi_boundary_from_url()
    st.session_state["delhi_gdf"] = gdf
    st.session_state["delhi_polygon"] = polygon

# --------- Data fetchers (unchanged)
@st.cache_data(ttl=600, show_spinner="Fetching Air Quality Data...")
def fetch_live_data():
    url = f"https://api.waqi.info/map/bounds/?latlng={DELHI_BOUNDS}&token={API_TOKEN}"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "ok":
            df = pd.DataFrame(data["data"])
            df = df[df['aqi'] != "-"]
            df['aqi'] = pd.to_numeric(df['aqi'], errors='coerce')
            df = df.dropna(subset=['aqi'])
            # Robust extract
            def safe_get_name(x):
                if isinstance(x, dict):
                    return x.get('name', 'N/A')
                elif isinstance(x, str):
                    return x
                else:
                    return 'N/A'
            def safe_get_time(x):
                if isinstance(x, dict):
                    time_data = x.get('time', {})
                    if isinstance(time_data, dict):
                        return time_data.get('s', 'N/A')
                    elif isinstance(time_data, str):
                        return time_data
                    else:
                        return 'N/A'
                else:
                    return 'N/A'
            df['station_name'] = df['station'].apply(safe_get_name)
            df['last_updated'] = df['station'].apply(safe_get_time)
            df[['category', 'color', 'emoji', 'advice']] = df['aqi'].apply(get_aqi_category).apply(pd.Series)
            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
            df = df.dropna(subset=['lat', 'lon'])
            return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=1800, show_spinner="Fetching Weather Data...")
def fetch_weather_data():
    url = f"https://api.open-meteo.com/v1/forecast?latitude={DELHI_LAT}&longitude={DELHI_LON}&current_weather=true&timezone=Asia/Kolkata"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

# -------------------------
# AQI category -- fixed hazardous color to black circle
# -------------------------
def get_aqi_category(aqi):
    """Categorizes AQI value and provides color, emoji, and health advice."""
    a = float(aqi)
    if a <= 50:
        return "Good", [0, 158, 96], "✅", "Enjoy outdoor activities."
    elif a <= 100:
        return "Moderate", [255, 214, 0], "🟡", "Unusually sensitive people should consider reducing prolonged or heavy exertion."
    elif a <= 150:
        return "Unhealthy for Sensitive Groups", [249, 115, 22], "🟠", "Sensitive groups should reduce prolonged or heavy exertion."
    elif a <= 200:
        return "Unhealthy", [220, 38, 38], "🔴", "Everyone may begin to experience health effects."
    elif a <= 300:
        return "Very Unhealthy", [147, 51, 234], "🟣", "Health alert: everyone may experience more serious health effects."
    else:
        # Hazardous: use black circle emoji and black RGB for UI markers
        return "Hazardous", [0, 0, 0], "⚫", "Health warnings of emergency conditions. The entire population is more likely to be affected."

# -------------------------
# Remaining UI and functions (kept from your original script)
# -------------------------
# For brevity I kept most of your existing UI/visual code identical; only the SMS send call in the subscription tab is replaced
# We'll include the key rendering functions: header/map/alerts/kriging/subscription etc.
# (I will re-use your existing functions with minimal edits where required.)

# --- render_header (shortened to essential content)
def render_header(df):
    st.markdown('<div style="font-size:2.2rem; font-weight:800; text-align:center;">🌍 Delhi Air Quality Dashboard</div>', unsafe_allow_html=True)
    last_update_time = df['last_updated'].max() if not df.empty and 'last_updated' in df.columns else "N/A"
    st.markdown(f'<div style="text-align:center; color: #555;">Last updated: {last_update_time}</div>', unsafe_allow_html=True)
    if not df.empty:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Average AQI", f"{df['aqi'].mean():.1f}")
        with c2:
            st.metric("Min AQI", f"{df['aqi'].min():.0f}", df.loc[df['aqi'].idxmin()]['station_name'])
        with c3:
            st.metric("Max AQI", f"{df['aqi'].max():.0f}", df.loc[df['aqi'].idxmax()]['station_name'])


def render_map_tab(df):
    st.subheader("📍 Live Map")
    if df.empty:
        st.warning("No data")
        return
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=DELHI_LAT, longitude=DELHI_LON, zoom=9.5),
        layers=[pdk.Layer("ScatterplotLayer", data=df, get_position='[lon, lat]', get_fill_color='color', get_radius=250)],
        tooltip={"html": "<b>{station_name}</b><br/>AQI: {aqi}", "style": {"color": "white"}}
    ))

def render_alerts_tab(df):
    st.subheader("🔔 Alerts & Health Advice")
    if df.empty:
        st.info("No station data.")
        return
    max_aqi = df['aqi'].max()
    _, _, emoji, advice = get_aqi_category(max_aqi)
    st.info(f"{emoji} Current highest AQI: {max_aqi:.0f} — {advice}")

# Kriging tab (kept as-is but with guard)
def render_kriging_tab(df):
    st.subheader("🔥 Kriging Heatmap (Spatial Interpolation)")
    if df.empty or len(df) < 3:
        st.warning("Not enough stations for kriging.")
        return
    try:
        delhi_polygon = st.session_state.get("delhi_polygon", None)
        lon_grid, lat_grid, z = perform_kriging_correct(df, (28.40, 28.88, 76.84, 77.35), polygon=delhi_polygon, resolution=250)
        st.session_state["kriging_output"] = (lon_grid, lat_grid, z)
        heatmap_df = pd.DataFrame({"lon": lon_grid.flatten(), "lat": lat_grid.flatten(), "aqi": z.flatten()}).dropna()
        fig = px.density_mapbox(heatmap_df, lat="lat", lon="lon", z="aqi", radius=12, center=dict(lat=DELHI_LAT, lon=DELHI_LON), zoom=9.5, mapbox_style="carto-positron", color_continuous_scale=["#009E60","#FFD600","#F97316","#DC2626","#9333EA","#000000"], range_color=[0,500])
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Kriging failed: {str(e)}")
        st.code(traceback.format_exc())

# -------------------------
# Subscription / SMS Alerts tab (FIXED)
# -------------------------
def render_alert_subscription_tab(df):
    st.subheader("📩 Real-Time AQI Alerts (via SMS)")

    polygon = st.session_state.get("delhi_polygon", None)
    if polygon is None:
        st.error("Delhi polygon not loaded.")
        return

    # Ensure kriging data exists; generate if needed
    kriging_data = st.session_state.get("kriging_output", None)
    if kriging_data is None:
        if df.empty or len(df) < 3:
            st.error("Not enough stations to compute interpolated AQI for alerts (min 3 required).")
            return
        with st.spinner("Generating interpolation for alerts..."):
            try:
                lon_grid, lat_grid, z_grid = perform_kriging_correct(df, (28.40, 28.88, 76.84, 77.35), polygon=polygon, resolution=200)
                st.session_state["kriging_output"] = (lon_grid, lat_grid, z_grid)
                kriging_data = (lon_grid, lat_grid, z_grid)
            except Exception as e:
                st.error("Could not create kriging output.")
                st.code(traceback.format_exc())
                return

    lon_grid, lat_grid, z_grid = kriging_data

    st.markdown("### Select notification location")
    mode = st.radio("How to specify location:", ["Select from list", "Enter coordinates"], horizontal=True)
    user_lat = None
    user_lon = None
    if mode == "Select from list":
        presets = {
            "Connaught Place": (28.6315, 77.2167),
            "India Gate": (28.6129, 77.2295),
            "Karol Bagh": (28.6519, 77.1906),
            "Rohini": (28.7496, 77.0670),
            "Dwarka": (28.5921, 77.0460)
        }
        # Add monitoring stations
        station_choices = {f"{r['station_name']} (AQI {r['aqi']:.0f})": (r['lat'], r['lon']) for _, r in df.iterrows()}
        all_choices = {**presets, **station_choices}
        choice = st.selectbox("Choose location", options=list(all_choices.keys()))
        user_lat, user_lon = all_choices[choice]
        st.write("Selected:", choice, f"({user_lat:.4f}, {user_lon:.4f})")
    else:
        c1, c2 = st.columns(2)
        with c1:
            user_lat = st.number_input("Latitude", value=28.6139, format="%.6f")
        with c2:
            user_lon = st.number_input("Longitude", value=77.2090, format="%.6f")

    st.markdown("---")
    phone_input = st.text_input("Phone number (E.164, e.g. +919876543210)")
    notify_btn = st.button("Send AQI Alert SMS")

    if notify_btn:
        # Validate phone and location
        if not phone_input:
            st.warning("Please provide phone number.")
            return
        if not validate_phone_number(phone_input):
            st.warning("Invalid phone format. Use +<countrycode><number>, digits only.")
            return
        if user_lat is None or user_lon is None:
            st.warning("Provide coordinates or select a location.")
            return

        # Get interpolated AQI for user location
        try:
            aqi_val, outside = get_aqi_at_location(user_lat, user_lon, lat_grid, lon_grid, z_grid, polygon)
        except Exception as e:
            st.error("Failed to compute interpolated AQI.")
            st.code(traceback.format_exc())
            return

        if np.isnan(aqi_val):
            st.error("Could not determine AQI for this location. Try another location.")
            return

        # Compose message
        category, color_rgb, emoji, advice = get_aqi_category(aqi_val)
        weather_json = fetch_weather_data()
        if weather_json and "current_weather" in weather_json:
            weather_desc = weather_json["current_weather"].get("weathercode", "N/A")
            temp = weather_json["current_weather"].get("temperature", "N/A")
        else:
            weather_desc = "N/A"
            temp = "N/A"

        # build human-friendly message (short)
        send_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        message = (f"Delhi AQI Alert ({send_time})\n"
                   f"Location: {user_lat:.4f}, {user_lon:.4f}\n"
                   f"{emoji} AQI: {aqi_val:.0f} ({category})\n"
                   f"Temp: {temp}°C • Weather: {weather_desc}\n"
                   f"Advice: {advice}\n"
                   f"Source: Interpolated AQI from local monitoring stations.")

        # Attempt to send SMS using wrapper
        with st.spinner("Sending SMS..."):
            success, provider, resp_text = send_sms(phone_input, message)
            if success:
                st.success(f"SMS sent successfully via {provider}.")
            else:
                # Detailed human-friendly error
                if provider == "validation":
                    st.error(f"Phone validation failed: {resp_text}")
                elif provider == "throttle":
                    st.warning(resp_text)
                elif provider in ("sms77_failed", "sms77", "both_failed", "twilio"):
                    st.warning(f"SMS sending failed ({provider}): {resp_text}")
                    st.info("Showing the alert here (SMS failed).")
                else:
                    st.error(f"SMS error ({provider}): {resp_text}")

        # Always display the alert on screen (even if SMS failed)
        st.markdown("---")
        st.markdown(f"### {emoji} Local AQI: {aqi_val:.0f} — {category}")
        st.markdown(f"**Advice:** {advice}")
        st.markdown(f"**Message Sent To:** {phone_input} (provider_result: {provider})")
        if provider and resp_text:
            st.code(str(resp_text))

# -------------------------
# Remaining tabs + main
# -------------------------
def render_dummy_forecast_tab():
    st.subheader("Forecast (Simulated)")
    hours = np.arange(0, 24)
    base_aqi = 120 + 40 * np.sin(hours / 3) + np.random.normal(0, 5, size=24)
    timestamps = [datetime.now() + timedelta(hours=i) for i in range(24)]
    forecast_df = pd.DataFrame({"timestamp": timestamps, "forecast_aqi": np.clip(base_aqi, 40, 300)})
    fig = px.line(forecast_df, x="timestamp", y="forecast_aqi", title="24h Forecast (Simulated)", markers=True)
    st.plotly_chart(fig, use_container_width=True)

def render_analytics_tab(df):
    st.subheader("Analytics")
    if df.empty:
        st.info("No data")
        return
    fig = px.pie(values=df['aqi'].value_counts().values, names=df['aqi'].value_counts().index)
    st.plotly_chart(fig)

# -------------------------
# MAIN
# -------------------------
aqi_data_raw = fetch_live_data()

if aqi_data_raw.empty:
    st.error("Could not fetch live AQI data. Check API key or network.")
    render_header(aqi_data_raw)
else:
    # Clip to Delhi geometry if available
    delhi_polygon = st.session_state.get("delhi_polygon", None)
    aqi_display_df = aqi_data_raw
    if delhi_polygon is not None and not aqi_data_raw.empty:
        try:
            geometry = [Point(xy) for xy in zip(aqi_data_raw['lon'], aqi_data_raw['lat'])]
            stations_gdf = gpd.GeoDataFrame(aqi_data_raw, crs="epsg:4326", geometry=geometry)
            clipped = gpd.clip(stations_gdf, delhi_polygon)
            if not clipped.empty:
                aqi_display_df = pd.DataFrame(clipped.drop(columns='geometry'))
        except Exception:
            aqi_display_df = aqi_data_raw

    render_header(aqi_display_df)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Map", "Alerts", "Analytics", "SMS Alerts", "Forecast", "Kriging"])
    with tab1:
        render_map_tab(aqi_display_df)
    with tab2:
        render_alerts_tab(aqi_display_df)
    with tab3:
        render_analytics_tab(aqi_display_df)
    with tab4:
        render_alert_subscription_tab(aqi_display_df)
    with tab5:
        render_dummy_forecast_tab()
    with tab6:
        render_kriging_tab(aqi_display_df)
