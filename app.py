import streamlit as st
import pandas as pd
import numpy as np
import requests
import pydeck as pdk
import plotly.express as px
from datetime import datetime, timedelta

# ==========================
# PAGE CONFIGURATION
# ==========================
st.set_page_config(
    layout="wide",
    page_title="Delhi Air Quality Dashboard",
    page_icon="💨"
)

# ==========================
# STATIC CONFIG
# ==========================
API_TOKEN = "97a0e712f47007556b57ab4b14843e72b416c0f9"
DELHI_BOUNDS = "28.404,76.840,28.883,77.349"
DELHI_LAT = 28.6139
DELHI_LON = 77.2090

# Twilio Configuration (you need to add your credentials)
TWILIO_ACCOUNT_SID = "AC2cc57109fc63de336609901187eca69d"
TWILIO_AUTH_TOKEN = "62b791789bb490f91879e89fa2ed959d"
TWILIO_PHONE_NUMBER = "+13856005348"

# ==========================
# CUSTOM CSS FOR STYLING
# ==========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background - Sky Blue Theme */
    .stApp {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 50%, #90CAF9 100%);
    }

    /* Hide Streamlit's default header and footer */
    header, footer, #MainMenu {
        visibility: hidden;
    }
    
    /* Main title styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        color: #0D47A1;
        padding: 1.5rem 0 0.5rem 0;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(13, 71, 161, 0.2);
        letter-spacing: -1px;
    }

    /* Subtitle styling */
    .subtitle {
        font-size: 1.2rem;
        color: #1565C0;
        text-align: center;
        padding-bottom: 1.5rem;
        font-weight: 500;
    }

    /* Metric cards styling */
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 1.5rem;
        border: 2px solid #BBDEFB;
        box-shadow: 0 4px 20px rgba(33, 150, 243, 0.15);
        text-align: center;
        height: 100%;
    }
    .metric-card-label {
        font-size: 1rem;
        font-weight: 600;
        color: #1565C0;
        margin-bottom: 0.5rem;
    }
    .metric-card-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0D47A1;
        margin: 0.5rem 0;
    }
    .metric-card-delta {
        font-size: 0.9rem;
        color: #1976D2;
        font-weight: 500;
    }

    /* Weather widget styling */
    .weather-widget {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 1.5rem;
        border: 2px solid #BBDEFB;
        box-shadow: 0 4px 20px rgba(33, 150, 243, 0.15);
        height: 100%;
    }
    .weather-temp {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0D47A1;
    }

    /* Styling for Streamlit tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: transparent;
        padding: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        background-color: white;
        border-radius: 15px;
        padding: 1rem 2rem;
        border: 2px solid #BBDEFB;
        color: #1565C0;
        box-shadow: 0 2px 10px rgba(33, 150, 243, 0.1);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #E3F2FD;
        border-color: #2196F3;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        color: white !important;
        border-color: #1976D2;
    }

    /* General card for content */
    .content-card {
        background-color: #FFFFFF;
        padding: 2rem;
        border-radius: 20px;
        border: 2px solid #BBDEFB;
        box-shadow: 0 10px 40px rgba(33, 150, 243, 0.2);
        margin-top: 1.5rem;
    }

    /* Alert cards for different severity levels */
    .alert-card {
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: white;
        font-weight: 600;
    }
    .alert-hazardous { 
        background: linear-gradient(135deg, #EF5350 0%, #E53935 100%);
        box-shadow: 0 4px 15px rgba(239, 83, 80, 0.3);
    }
    .alert-very-unhealthy { 
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
        box-shadow: 0 4px 15px rgba(255, 152, 0, 0.3);
    }
    .alert-unhealthy { 
        background: linear-gradient(135deg, #FFA726 0%, #FB8C00 100%);
        box-shadow: 0 4px 15px rgba(255, 167, 38, 0.3);
    }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0D47A1;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #BBDEFB;
    }

    /* Info box styling */
    div[data-testid="stAlert"] {
        background-color: white;
        border-left: 5px solid #2196F3;
        border-radius: 10px;
        color: #0D47A1;
    }

    /* Success box styling */
    div[data-testid="stSuccess"] {
        background-color: white;
        border-left: 5px solid #4CAF50;
        border-radius: 10px;
        color: #2E7D32;
    }

    /* Error box styling */
    div[data-testid="stError"] {
        background-color: white;
        border-left: 5px solid #EF5350;
        border-radius: 10px;
        color: #C62828;
    }

    /* Dataframe styling */
    div[data-testid="stDataFrame"] {
        border: 2px solid #BBDEFB;
        border-radius: 10px;
        background-color: white;
    }
    
    /* Chart containers */
    div[data-testid="stPlotlyChart"] {
        background-color: white;
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    /* Ensure all containers have white background */
    .element-container {
        background-color: transparent;
    }
    
    /* Block container styling */
    .block-container {
        background-color: transparent;
        padding-top: 2rem;
    }

</style>
""", unsafe_allow_html=True)

# ==========================
# HELPER FUNCTIONS
# ==========================


@st.cache_data(ttl=600, show_spinner="Fetching Air Quality Data...")
def fetch_live_data():
    """Fetches and processes live AQI data from the WAQI API."""
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
            # Robustly extract station name and last updated time

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
            df[['category', 'color', 'emoji', 'advice']] = df['aqi'].apply(
                get_aqi_category).apply(pd.Series)
            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
            df = df.dropna(subset=['lat', 'lon'])
            return df
        return pd.DataFrame()
    except requests.RequestException:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner="Fetching Weather Data...")
def fetch_weather_data():
    """Fetches current weather data from Open-Meteo API."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={DELHI_LAT}&longitude={DELHI_LON}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&timezone=Asia/Kolkata"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def get_aqi_category(aqi):
    """Categorizes AQI value and provides color, emoji, and health advice."""
    if aqi <= 50:
        return "Good", [0, 158, 96], "✅", "Enjoy outdoor activities."
    elif aqi <= 100:
        return "Moderate", [255, 214, 0], "🟡", "Unusually sensitive people should consider reducing prolonged or heavy exertion."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", [249, 115, 22], "🟠", "Sensitive groups should reduce prolonged or heavy exertion."
    elif aqi <= 200:
        return "Unhealthy", [220, 38, 38], "🔴", "Everyone may begin to experience health effects."
    elif aqi <= 300:
        return "Very Unhealthy", [147, 51, 234], "🟣", "Health alert: everyone may experience more serious health effects."
    else:
        return "Hazardous", [126, 34, 206], "☠️", "Health warnings of emergency conditions. The entire population is more likely to be affected."


def get_weather_info(code):
    """Converts WMO weather code to a description and icon."""
    codes = {
        0: ("Clear sky", "☀️"), 1: ("Mainly clear", "🌤️"), 2: ("Partly cloudy", "⛅"),
        3: ("Overcast", "☁️"), 45: ("Fog", "🌫️"), 48: ("Depositing rime fog", "🌫️"),
        51: ("Light drizzle", "💧"), 53: ("Moderate drizzle", "💧"), 55: ("Dense drizzle", "💧"),
        61: ("Slight rain", "🌧️"), 63: ("Moderate rain", "🌧️"), 65: ("Heavy rain", "🌧️"),
        80: ("Slight rain showers", "🌦️"), 81: ("Moderate rain showers", "🌦️"),
        82: ("Violent rain showers", "⛈️"), 95: ("Thunderstorm", "⚡"),
        96: ("Thunderstorm, slight hail", "⛈️"), 99: ("Thunderstorm, heavy hail", "⛈️")
    }
    return codes.get(code, ("Unknown", "❓"))


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates using Haversine formula."""
    from math import radians, sin, cos, sqrt, atan2

    R = 6371  # Earth's radius in kilometers

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c

    return distance


def get_nearby_stations(df, user_lat, user_lon, radius_km=10):
    """Get stations within specified radius of user location."""
    df['distance'] = df.apply(
        lambda row: calculate_distance(
            user_lat, user_lon, row['lat'], row['lon']),
        axis=1
    )
    nearby = df[df['distance'] <= radius_km].sort_values('distance')
    return nearby


def send_sms_alert(phone_number, message):
    """Send SMS alert using Twilio."""
    try:
        from twilio.rest import Client

        # Check if credentials are configured
        if TWILIO_ACCOUNT_SID == "your_twilio_account_sid" or not TWILIO_ACCOUNT_SID.startswith("AC"):
            return False, "⚠️ Twilio Account SID not configured correctly. It should start with 'AC' and be 34 characters long."

        if TWILIO_AUTH_TOKEN == "your_twilio_auth_token" or len(TWILIO_AUTH_TOKEN) < 30:
            return False, "⚠️ Twilio Auth Token not configured correctly. It should be 32 characters long."

        if TWILIO_PHONE_NUMBER == "your_twilio_phone_number" or not TWILIO_PHONE_NUMBER.startswith("+"):
            return False, "⚠️ Twilio Phone Number not configured correctly. It should start with '+' and include country code."

        # Validate recipient phone number
        if not phone_number.startswith("+"):
            return False, "⚠️ Recipient phone number must include country code starting with '+'"

        # Create Twilio client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        # Send message
        sent_message = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=phone_number
        )

        return True, f"✅ Alert sent successfully! Message SID: {sent_message.sid}"
    except ImportError:
        return False, "❌ Twilio library not installed. Run: pip install twilio"
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "authenticate" in error_msg.lower():
            return False, f"🔐 Authentication Error: Your Twilio credentials are incorrect.\n\n✓ Check Account SID (starts with 'AC')\n✓ Check Auth Token (click eye icon 👁️ in console to reveal)\n✓ Make sure there are no extra spaces\n\nError details: {error_msg}"
        elif "unverified" in error_msg.lower():
            return False, f"📱 Phone Number Not Verified: For trial accounts, you must verify the recipient number in Twilio Console.\n\nGo to: https://console.twilio.com/us1/develop/phone-numbers/manage/verified\n\nError details: {error_msg}"
        else:
            return False, f"❌ Error sending SMS: {error_msg}"


def create_alert_message(nearby_stations, weather_data, location_name):
    """Create alert message with AQI and weather information."""
    if nearby_stations.empty:
        return "No nearby air quality monitoring stations found."

    # Get average AQI and worst station
    avg_aqi = nearby_stations['aqi'].mean()
    worst_station = nearby_stations.iloc[0]

    # Get weather info
    weather_desc = "N/A"
    temp = "N/A"
    if weather_data and 'current' in weather_data:
        current = weather_data['current']
        weather_desc, _ = get_weather_info(current.get('weather_code', 0))
        temp = f"{current['temperature_2m']:.1f}°C"

    # Create message
    category, _, emoji, advice = get_aqi_category(avg_aqi)

    message = f"""🌍 Air Quality Alert - {location_name}

{emoji} AQI Status: {category}
📊 Average AQI: {avg_aqi:.0f}

🔴 Worst Station: {worst_station['station_name']}
AQI: {worst_station['aqi']:.0f} ({worst_station['distance']:.1f} km away)

🌤️ Weather: {weather_desc}
🌡️ Temperature: {temp}

💡 Advice: {advice}

Stay safe!"""

    return message

# ==========================
# UI RENDERING FUNCTIONS
# ==========================


def render_header(df):
    """Renders the main header with summary metrics and weather."""
    st.markdown('<div class="main-title">🌍 Delhi Air Quality Dashboard</div>',
                unsafe_allow_html=True)
    last_update_time = df['last_updated'].max(
    ) if not df.empty and 'last_updated' in df.columns else "N/A"
    st.markdown(
        f'<p class="subtitle">Real-time monitoring • Last updated: {last_update_time}</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    if not df.empty:
        with c1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-card-label">Average AQI</div><div class="metric-card-value">{df["aqi"].mean():.1f}</div><div class="metric-card-delta">{get_aqi_category(df["aqi"].mean())[0]}</div></div>', unsafe_allow_html=True)
        with c2:
            min_station = df.loc[df["aqi"].idxmin()]["station_name"]
            st.markdown(
                f'<div class="metric-card"><div class="metric-card-label">Minimum AQI</div><div class="metric-card-value">{df["aqi"].min():.0f}</div><div class="metric-card-delta">{min_station}</div></div>', unsafe_allow_html=True)
        with c3:
            max_station = df.loc[df["aqi"].idxmax()]["station_name"]
            st.markdown(
                f'<div class="metric-card"><div class="metric-card-label">Maximum AQI</div><div class="metric-card-value">{df["aqi"].max():.0f}</div><div class="metric-card-delta">{max_station}</div></div>', unsafe_allow_html=True)

    with c4:
        weather_data = fetch_weather_data()
        if weather_data and 'current' in weather_data:
            current = weather_data['current']
            desc, icon = get_weather_info(current.get('weather_code', 0))
            st.markdown(f"""
            <div class="weather-widget">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div>
                        <div class="metric-card-label">Current Weather</div>
                        <div class="weather-temp">{current['temperature_2m']:.1f}°C</div>
                    </div>
                    <div style="font-size: 3rem;">{icon}</div>
                </div>
                <div style="text-align: left; font-size: 0.9rem; color: #1976D2; margin-top: 1rem; font-weight: 500;">
                    {desc}<br/>Humidity: {current['relative_humidity_2m']}%<br/>Wind: {current['wind_speed_10m']} km/h
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="weather-widget">
                <div class="metric-card-label">Current Weather</div>
                <div style="color: #1976D2; margin-top: 1rem;">Weather data unavailable</div>
            </div>
            """, unsafe_allow_html=True)


def render_map_tab(df):
    """Renders the interactive map of AQI stations."""
    st.markdown('<div class="section-header">📍 Interactive Air Quality Map</div>',
                unsafe_allow_html=True)

    # Add Legend
    st.markdown("""
    <div style="background-color: white; padding: 1rem; border-radius: 10px; border: 2px solid #BBDEFB; margin-bottom: 1rem;">
        <div style="font-weight: 700; color: #0D47A1; margin-bottom: 0.75rem; font-size: 1.1rem;">AQI Color Legend</div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <div style="width: 20px; height: 20px; border-radius: 50%; background-color: rgb(0, 158, 96);"></div>
                <span style="color: #1E293B; font-weight: 500;">Good (0-50)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <div style="width: 20px; height: 20px; border-radius: 50%; background-color: rgb(255, 214, 0);"></div>
                <span style="color: #1E293B; font-weight: 500;">Moderate (51-100)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <div style="width: 20px; height: 20px; border-radius: 50%; background-color: rgb(249, 115, 22);"></div>
                <span style="color: #1E293B; font-weight: 500;">Unhealthy for Sensitive (101-150)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <div style="width: 20px; height: 20px; border-radius: 50%; background-color: rgb(220, 38, 38);"></div>
                <span style="color: #1E293B; font-weight: 500;">Unhealthy (151-200)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <div style="width: 20px; height: 20px; border-radius: 50%; background-color: rgb(147, 51, 234);"></div>
                <span style="color: #1E293B; font-weight: 500;">Very Unhealthy (201-300)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <div style="width: 20px; height: 20px; border-radius: 50%; background-color: rgb(126, 34, 206);"></div>
                <span style="color: #1E293B; font-weight: 500;">Hazardous (300+)</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.pydeck_chart(pdk.Deck(
        map_style="light",
        initial_view_state=pdk.ViewState(
            latitude=DELHI_LAT, longitude=DELHI_LON, zoom=9.5, pitch=50),
        layers=[pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position='[lon, lat]',
            get_fill_color='color',
            get_radius=250,
            pickable=True,
            opacity=0.8,
            stroked=True,
            get_line_color=[0, 0, 0, 100],
            line_width_min_pixels=1,
        )],
        tooltip={"html": "<b>{station_name}</b><br/>AQI: {aqi}<br/>Category: {category}<br/>Last Updated: {last_updated}",
                 "style": {"color": "white"}}
    ))


def render_alerts_tab(df):
    """Renders health alerts and advice based on current AQI levels."""
    st.markdown('<div class="section-header">🔔 Health Alerts & Recommendations</div>',
                unsafe_allow_html=True)
    max_aqi = df['aqi'].max()
    advice = get_aqi_category(max_aqi)[3]
    st.info(
        f"**Current Situation:** Based on the highest AQI of **{max_aqi:.0f}**, the recommended action is: **{advice}**", icon="ℹ️")

    alerts = {
        "Hazardous": (df[df['aqi'] > 300], "alert-hazardous"),
        "Very Unhealthy": (df[(df['aqi'] > 200) & (df['aqi'] <= 300)], "alert-very-unhealthy"),
        "Unhealthy": (df[(df['aqi'] > 150) & (df['aqi'] <= 200)], "alert-unhealthy")
    }
    has_alerts = False
    for level, (subset, card_class) in alerts.items():
        if not subset.empty:
            has_alerts = True
            st.markdown(
                f"**{subset.iloc[0]['emoji']} {level} Conditions Detected**")
            for _, row in subset.sort_values('aqi', ascending=False).iterrows():
                st.markdown(
                    f'<div class="alert-card {card_class}"><span style="font-weight: 600;">{row["station_name"]}</span> <span style="font-weight: 700; font-size: 1.2rem;">AQI {row["aqi"]:.0f}</span></div>', unsafe_allow_html=True)

    if not has_alerts:
        st.success("✅ No significant air quality alerts at the moment. AQI levels are currently within the good to moderate range for most areas.", icon="✅")


def render_alert_subscription_tab(df):
    """Renders alert subscription form."""
    st.markdown('<div class="section-header">📱 SMS Alert Subscription</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #E3F2FD; padding: 1rem; border-radius: 10px; border-left: 4px solid #2196F3; margin-bottom: 1.5rem;">
        <p style="color: #0D47A1; margin: 0; font-weight: 500;">
        📍 Get real-time air quality and weather alerts for your location via SMS. 
        We'll find the nearest monitoring stations and send you personalized updates.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        location_name = st.text_input(
            "📍 Your Location Name",
            placeholder="e.g., Connaught Place, New Delhi",
            help="Enter your area/locality name"
        )

        user_lat = st.number_input(
            "Latitude",
            min_value=28.4,
            max_value=28.9,
            value=28.6139,
            step=0.0001,
            format="%.4f",
            help="Your location's latitude"
        )

        user_lon = st.number_input(
            "Longitude",
            min_value=76.8,
            max_value=77.4,
            value=77.2090,
            step=0.0001,
            format="%.4f",
            help="Your location's longitude"
        )

    with col2:
        phone_number = st.text_input(
            "📱 Phone Number",
            placeholder="+91XXXXXXXXXX",
            help="Enter with country code (e.g., +919876543210)"
        )

        radius = st.slider(
            "Search Radius (km)",
            min_value=1,
            max_value=20,
            value=10,
            help="Find stations within this radius"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        send_alert_btn = st.button(
            "📤 Send Alert Now", type="primary", use_container_width=True)

    if send_alert_btn:
        if not location_name or not phone_number:
            st.error(
                "Please fill in all required fields: Location Name and Phone Number", icon="⚠️")
        elif not phone_number.startswith('+'):
            st.error(
                "Phone number must include country code (e.g., +919876543210)", icon="⚠️")
        else:
            with st.spinner("Finding nearby stations and preparing alert..."):
                # Get nearby stations
                nearby_stations = get_nearby_stations(
                    df, user_lat, user_lon, radius)

                if nearby_stations.empty:
                    st.warning(
                        f"No monitoring stations found within {radius} km of your location. Try increasing the search radius.", icon="⚠️")
                else:
                    # Get weather data
                    weather_data = fetch_weather_data()

                    # Create alert message
                    alert_message = create_alert_message(
                        nearby_stations, weather_data, location_name)

                    # Display preview
                    st.markdown("### 📄 Alert Preview")
                    st.info(alert_message)

                    # Show nearby stations
                    st.markdown("### 📍 Nearby Monitoring Stations")
                    display_nearby = nearby_stations[[
                        'station_name', 'aqi', 'category', 'distance']].head(5)
                    display_nearby['distance'] = display_nearby['distance'].round(
                        2).astype(str) + ' km'
                    st.dataframe(display_nearby,
                                 use_container_width=True, hide_index=True)

                    # Send SMS
                    success, message = send_sms_alert(
                        phone_number, alert_message)

                    if success:
                        st.success(message, icon="✅")
                    else:
                        st.error(message, icon="❌")
                        st.info("💡 **Note:** To enable SMS alerts, you need to:\n1. Sign up for Twilio (free trial available)\n2. Get your Account SID, Auth Token, and Phone Number\n3. Update the configuration in the code\n4. Install Twilio: `pip install twilio`", icon="ℹ️")


def render_dummy_forecast_tab():
    """Render a dummy 24-hour AQI forecast using simulated data."""
    st.markdown('<div class="section-header">📈 24-Hour AQI Forecast (Sample)</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #E3F2FD; padding: 1rem; border-radius: 10px; border-left: 4px solid #2196F3; margin-bottom: 1rem;">
        <p style="color: #0D47A1; margin: 0; font-weight: 500;">
        This sample forecast simulates how the Air Quality Index (AQI) may change over the next 24 hours.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Simulate a smooth AQI forecast for 24 hours
    hours = np.arange(0, 24)
    base_aqi = 120 + 40 * np.sin(hours / 3) + np.random.normal(0, 5, size=24)
    timestamps = [datetime.now() + timedelta(hours=i) for i in range(24)]
    forecast_df = pd.DataFrame({
        "timestamp": timestamps,
        "forecast_aqi": np.clip(base_aqi, 40, 300)
    })

    # Plot forecast trend
    fig = px.line(
        forecast_df,
        x="timestamp",
        y="forecast_aqi",
        title="Predicted AQI Trend for Next 24 Hours (Simulated)",
        markers=True,
        line_shape="spline"
    )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Predicted AQI",
        showlegend=False,
        margin=dict(t=40, b=20, l=0, r=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
        title_font_color="#0D47A1",
        font_color="#0D47A1",
        xaxis=dict(gridcolor='#E3F2FD'),
        yaxis=dict(gridcolor='#E3F2FD')
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display summary
    avg_aqi = forecast_df["forecast_aqi"].mean()
    max_aqi = forecast_df["forecast_aqi"].max()
    min_aqi = forecast_df["forecast_aqi"].min()

    st.markdown(f"""
    <div style="background-color: white; padding: 1rem; border-radius: 10px; border-left: 5px solid #1976D2; margin-top: 1rem; color: #1E293B;">
        <b>Average Forecasted AQI:</b> {avg_aqi:.1f}  
        <br><b>Expected Range:</b> {min_aqi:.1f} – {max_aqi:.1f}
        <br><b>Air Quality Outlook:</b> Moderate to Unhealthy range over the next day.
    </div>
    """, unsafe_allow_html=True)

def render_analytics_tab(df):
    """Renders charts and data analytics."""
    st.markdown('<div class="section-header">📊 Data Analytics</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("**AQI Category Distribution**")
        category_counts = df['category'].value_counts()
        fig = px.pie(
            values=category_counts.values, names=category_counts.index, hole=0.4,
            color=category_counts.index,
            color_discrete_map={
                "Good": "#009E60", "Moderate": "#FFD600", "Unhealthy for Sensitive Groups": "#F97316",
                "Unhealthy": "#DC2626", "Very Unhealthy": "#9333EA", "Hazardous": "#7E22CE"
            }
        )
        fig.update_traces(textinfo='percent+label',
                          pull=[0.05]*len(category_counts.index))
        fig.update_layout(
            showlegend=False,
            margin=dict(t=0, b=0, l=0, r=0),
            paper_bgcolor='#F5F5F5',
            plot_bgcolor='#F5F5F5'
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Top 10 Most Polluted Stations**")
        top_10 = df.nlargest(10, 'aqi').sort_values('aqi', ascending=True)
        fig = px.bar(
            top_10, x='aqi', y='station_name', orientation='h',
            color='aqi', color_continuous_scale=px.colors.sequential.Reds
        )
        fig.update_layout(
            xaxis_title="AQI",
            yaxis_title="",
            showlegend=False,
            margin=dict(t=20, b=20, l=0, r=20),
            paper_bgcolor='#F5F5F5',
            plot_bgcolor='#F5F5F5',
            xaxis=dict(gridcolor='#DDDDDD'),
            yaxis=dict(gridcolor='#DDDDDD')
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Full Station Data**")
    display_df = df[['station_name', 'aqi', 'category',
                     'last_updated']].sort_values('aqi', ascending=False)
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ==========================
# MAIN APP EXECUTION
# ==========================
aqi_data = fetch_live_data()
render_header(aqi_data)

if aqi_data.empty:
    st.error("⚠️ **Could not fetch live AQI data.** The API may be down or there's a network issue. Please try again later.", icon="🚨")
else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🗺️ Live Map", "🔔 Alerts & Health", "📊 Analytics", "📱 SMS Alerts", "📈 Forecast"])
    with tab1:
        with st.container():
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            render_map_tab(aqi_data)
            st.markdown('</div>', unsafe_allow_html=True)
    with tab2:
        with st.container():
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            render_alerts_tab(aqi_data)
            st.markdown('</div>', unsafe_allow_html=True)
    with tab3:
        with st.container():
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            render_analytics_tab(aqi_data)
            st.markdown('</div>', unsafe_allow_html=True)
    with tab4:
        with st.container():
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            render_alert_subscription_tab(aqi_data)
            st.markdown('</div>', unsafe_allow_html=True)
    with tab5:
        with st.container():
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            render_dummy_forecast_tab()
            st.markdown('</div>', unsafe_allow_html=True)
# ======= BEGIN: CONTINUATION CODE (paste into app.py) =======

import math
import json
from urllib.parse import urlencode

# Optional: only required if you want to enable LSTM model usage
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

# --------------------------
# OpenAQ: fetch latest station-level data for Delhi
# --------------------------
@st.cache_data(ttl=300, show_spinner="Fetching OpenAQ data...")
def fetch_openaq_latest(city="Delhi", limit=1000):
    """Fetch latest measurements from OpenAQ for a city and return DataFrame matching app schema."""
    base = "https://api.openaq.org/v2/latest"
    params = {"city": city, "limit": limit}
    url = f"{base}?{urlencode(params)}"
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        payload = r.json()
        results = payload.get("results", [])
        rows = []
        for res in results:
            # each result may have multiple measurements; choose pm25 if available
            coords = res.get("coordinates", {})
            lat = coords.get("latitude")
            lon = coords.get("longitude")
            location = res.get("location") or res.get("name") or "N/A"
            # find pm25 measurement
            pm25 = None
            last_updated = None
            for m in res.get("measurements", []):
                if m.get("parameter") == "pm25":
                    pm25 = m.get("value")
                    last_updated = m.get("lastUpdated")
                    break
            # fallback to any measurement (e.g., pm10) as rough proxy
            if pm25 is None and res.get("measurements"):
                pm25 = res["measurements"][0].get("value")
                last_updated = res["measurements"][0].get("lastUpdated")
            if lat is None or lon is None or pm25 is None:
                continue
            # Convert PM2.5 to approximate AQI using US EPA breakpoint estimator (simple conversion).
            # NOTE: this is a rough conversion for demo. For accurate AQI, use station AQI when available.
            aqi_est = pm25_to_aqi(pm25)
            rows.append({
                "station": {"name": location, "time": {"s": last_updated}},
                "station_name": location,
                "lat": lat,
                "lon": lon,
                "aqi": aqi_est,
                "last_updated": last_updated
            })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df['aqi'] = pd.to_numeric(df['aqi'], errors='coerce')
        df = df.dropna(subset=['aqi', 'lat', 'lon'])
        df[['category', 'color', 'emoji', 'advice']] = df['aqi'].apply(get_aqi_category).apply(pd.Series)
        return df
    except requests.RequestException:
        return pd.DataFrame()

# --------------------------
# PM2.5 -> AQI rough converter (US EPA) - for demo only
# --------------------------
def pm25_to_aqi(pm25_val):
    """
    Convert PM2.5 concentration (μg/m³) to US EPA AQI using breakpoints.
    This is a simplified converter for demo; for exact AQI use official CPCB/waqi station AQI values.
    """
    try:
        c = float(pm25_val)
    except Exception:
        return np.nan

    # US EPA breakpoints (concise)
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for (Clow, Chigh, Ilow, Ihigh) in breakpoints:
        if Clow <= c <= Chigh:
            aqi = ((Ihigh - Ilow) / (Chigh - Clow)) * (c - Clow) + Ilow
            return round(aqi)
    # outside range
    return 500

# --------------------------
# NASA FIRMS: fetch recent fire hotspots count in a bounding box
# --------------------------
@st.cache_data(ttl=900, show_spinner="Fetching NASA FIRMS hotspots...")
def fetch_firms_count(minlat=27.5, minlon=74.5, maxlat=31.5, maxlon=78.5, hours=24):
    """
    Count fire hotspots in region in last `hours`.
    Default box covers Punjab/Haryana/Northwest UP area (adjust as needed).
    Returns integer count of hotspots in the interval.
    """
    # FIRMS provides CSV/KML endpoints; use VIIRS NRT 24h JSON-ish CSV endpoint via download
    # We'll use the Viirs NRT 24h CSV (public) — if rate limits, adjust.
    try:
        # Using the simpler CSV endpoint for NRT VIIRS (24h)
        # NOTE: FIRMS requires email when using some endpoints; here we use the public CSV.
        url = f"https://firms.modaps.eosdis.nasa.gov/mapserver/wfs?service=wfs&request=GetFeature&typeName=viirs_viirsSnpp_NRT&outputFormat=application/json&viewparams=time:PT{hours}H"
        # The WFS endpoint usage / params differ by deployment; as a fallback, we can query the general VIIRS 24h CSV
        csv_url = f"https://firms.modaps.eosdis.nasa.gov/viirs/viirs_snpp_24h.csv"
        resp = requests.get(csv_url, timeout=15)
        resp.raise_for_status()
        text = resp.text
        # parse CSV into DataFrame
        df = pd.read_csv(pd.compat.StringIO(text))
        # Filter by bbox
        df_region = df[(df['latitude'] >= minlat) & (df['latitude'] <= maxlat) &
                       (df['longitude'] >= minlon) & (df['longitude'] <= maxlon)]
        return int(len(df_region))
    except Exception:
        # If FIRMS fails, return None (we'll treat as unknown)
        return None

# --------------------------
# Utility: get a single aggregate row of "current features" for simulator
# --------------------------
def prepare_current_features(df):
    """
    From AQI station DataFrame produce simple aggregated features:
      - avg_aqi: mean AQI across stations (float)
      - worst_aqi: max AQI (float)
      - fire_count: integer hotspots in last 24h (from FIRMS)
      - traffic_index: proxy (default 100) — you can replace with TomTom API if available
      - industry_index: proxy (default 50)
      - wind_speed: from fetch_weather_data
    These are **proxy features** for the simple linear simulator.
    """
    avg_aqi = float(df['aqi'].mean()) if not df.empty else np.nan
    worst_aqi = float(df['aqi'].max()) if not df.empty else np.nan
    fire_count = fetch_firms_count()  # may be None
    # simple traffic proxy: daytime based
    hour = datetime.now().hour
    # coarse heuristic: traffic higher during 7-10 and 17-20
    traffic_index = 120 if (7 <= hour <= 10 or 17 <= hour <= 20) else 80
    industry_index = 50  # placeholder; you can improve with industrial proxy
    weather = fetch_weather_data()
    wind_speed = None
    if weather and 'current' in weather:
        wind_speed = weather['current'].get('wind_speed_10m')
    return {
        "avg_aqi": avg_aqi,
        "worst_aqi": worst_aqi,
        "fire_count": (fire_count if fire_count is not None else 0),
        "traffic_index": traffic_index,
        "industry_index": industry_index,
        "wind_speed": (wind_speed if wind_speed is not None else 2.0)
    }

# --------------------------
# Linear sensitivity-based policy simulator (fast, transparent)
# --------------------------
def simulate_policy_linear(features, fire_reduction=0.0, traffic_reduction=0.0, industry_reduction=0.0):
    """
    A simple additive linear model that estimates AQI from proxies:
      AQI_est = base + a*fire_count + b*traffic_index + c*industry_index + d*wind_adjust
    These coefficients are toy/demo values — for hackathon, you can calibrate them on historical data.
    Returns (baseline_aqi, counterfactual_aqi, details_dict)
    """
    # coefficients (toy/demo) - calibrate on your historical dataset for better results
    coeffs = {"fire": 1.8, "traffic": 1.5, "industry": 0.6, "wind": -4.0, "intercept": 40.0}
    fire = features['fire_count']
    traffic = features['traffic_index']
    industry = features['industry_index']
    wind = features['wind_speed'] if features.get('wind_speed') is not None else 2.0

    baseline = coeffs['intercept'] + coeffs['fire']*fire + coeffs['traffic']*traffic + coeffs['industry']*industry + coeffs['wind']*wind

    # apply reductions
    fire_new = fire * (1 - float(fire_reduction))
    traffic_new = traffic * (1 - float(traffic_reduction))
    industry_new = industry * (1 - float(industry_reduction))

    counter = coeffs['intercept'] + coeffs['fire']*fire_new + coeffs['traffic']*traffic_new + coeffs['industry']*industry_new + coeffs['wind']*wind

    return max(0, baseline), max(0, counter), {
        "coeffs": coeffs,
        "inputs": {"fire": fire, "traffic": traffic, "industry": industry, "wind": wind},
        "inputs_new": {"fire": fire_new, "traffic": traffic_new, "industry": industry_new}
    }

# --------------------------
# (Optional) Example: how to call your LSTM model - TEMPLATE
# --------------------------
def simulate_policy_lstm_template(features, fire_reduction=0.0, traffic_reduction=0.0, model_path="/mnt/data/lstm_aqi_model.h5", scaler=None, seq_len=24):
    """
    TEMPLATE: Use this only if you trained an LSTM/TFT model and know the preprocessing pipeline.
    - scaler: sklearn-like scaler with transform() expecting a 2D array [N, features]
    - seq_len: number of past timesteps model expects
    This function demonstrates how to:
      1) Build a feature vector/sequence
      2) Modify inputs for counterfactual
      3) Call model.predict
    YOU MUST adapt this to your feature order & scaling used in training.
    """
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow not available; install tensorflow to use LSTM integration.")

    # Load model (cache outside in production)
    model = load_model(model_path)

    # EXAMPLE: build a single-row feature vector order:
    # [prev_aqi, fire_count, traffic_index, industry_index, wind_speed, humidity, temp]
    # You must replace with your real features & scaling!
    prev_aqi = features.get('avg_aqi', 100)
    base_feat = [prev_aqi, features['fire_count'], features['traffic_index'], features['industry_index'], features['wind_speed']]

    # Create a rolling sequence by repeating last observation seq_len times (if you don't have history)
    seq = np.tile(base_feat, (seq_len, 1))  # shape (seq_len, n_features)

    # Baseline prediction
    if scaler is not None:
        seq_scaled = scaler.transform(seq)
    else:
        seq_scaled = seq  # NOT scaled - likely wrong; only a template

    # model may expect shape (1, seq_len, n_features)
    baseline_pred = model.predict(seq_scaled.reshape(1, seq_len, seq.shape[1]))[0, 0]

    # Counterfactual: modify fire & traffic
    base_feat_cf = [prev_aqi, features['fire_count']*(1-fire_reduction), features['traffic_index']*(1-traffic_reduction), features['industry_index'], features['wind_speed']]
    seq_cf = np.tile(base_feat_cf, (seq_len, 1))
    if scaler is not None:
        seq_cf_scaled = scaler.transform(seq_cf)
    else:
        seq_cf_scaled = seq_cf

    counter_pred = model.predict(seq_cf_scaled.reshape(1, seq_len, seq_cf.shape[1]))[0, 0]

    return float(baseline_pred), float(counter_pred)

# --------------------------
# Add a new UI tab: Policy Simulator
# --------------------------
def render_policy_simulator_tab(aqi_df):
    st.markdown('<div class="section-header">🧪 Policy Simulator — What-if AQI Impact</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: #E3F2FD; padding: 0.75rem; border-radius: 10px; border-left: 4px solid #2196F3; margin-bottom: 1rem;">
        Use the sliders below to simulate reductions in major sources (stubble burning / fires, traffic, industry)
        and see the estimated AQI change. This uses a simple, transparent linear model for instant results.
    </div>
    """, unsafe_allow_html=True)

    # prepare current feature proxies
    features = prepare_current_features(aqi_df)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg AQI (current)", f"{features['avg_aqi']:.1f}" if not math.isnan(features['avg_aqi']) else "N/A")
        st.write("Fire hotspots (24h):", features['fire_count'])
    with col2:
        st.metric("Traffic index (proxy)", f"{features['traffic_index']}")
        st.metric("Industry index (proxy)", f"{features['industry_index']}")
    with col3:
        st.metric("Wind speed (m/s)", f"{features['wind_speed']:.1f}")

    st.markdown("### Adjust policy sliders")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        fire_red = st.slider("Reduce fires (stubble burning) %", min_value=0, max_value=100, value=20, step=5)
    with col_b:
        traffic_red = st.slider("Reduce traffic emissions %", min_value=0, max_value=100, value=40, step=5)
    with col_c:
        industry_red = st.slider("Reduce industry emissions %", min_value=0, max_value=100, value=10, step=5)

    # Run linear simulator
    baseline, counterfactual, details = simulate_policy_linear(features, fire_reduction=fire_red/100.0, traffic_reduction=traffic_red/100.0, industry_reduction=industry_red/100.0)

    # Present results
    st.markdown("### Simulation result (linear model)")
    st.markdown(f"**Peak/Estimated AQI:** {baseline:.0f} → **{counterfactual:.0f}**  (Δ = {baseline-counterfactual:.0f})")

    # Plot baseline vs counterfactual simple bar
    fig = px.bar(
        x=["Baseline (No policy)", "Counterfactual"],
        y=[baseline, counterfactual],
        color=["Baseline", "Counterfactual"],
        labels={"x": "", "y": "Estimated AQI"},
        title="Estimated AQI: Baseline vs Counterfactual"
    )
    fig.update_layout(showlegend=False, paper_bgcolor='white', plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

    # Show details
    st.markdown("#### Model details & assumptions")
    st.markdown(f"- Linear model coefficients (demo): {details['coeffs']}")
    st.markdown(f"- Original inputs: fires={details['inputs']['fire']}, traffic={details['inputs']['traffic']}, industry={details['inputs']['industry']}")
    st.markdown(f"- Modified inputs: fires={details['inputs_new']['fire']:.2f}, traffic={details['inputs_new']['traffic']:.2f}, industry={details['inputs_new']['industry']:.2f}")
    st.info("Note: This is a rapid, interpretable estimate. For production-quality counterfactuals, plug in a validated time-series model (LSTM/TFT) and the exact preprocessing pipeline used in training.", icon="ℹ️")

    # Optional LSTM button (template)
    if TENSORFLOW_AVAILABLE:
        if st.button("Run LSTM-based counterfactual (requires your scaler & correct feature order)"):
            with st.spinner("Running LSTM template..."):
                try:
                    # Example: the function simulate_policy_lstm_template shows how to call your model.
                    # If you have a scaler object saved (sklearn) load it and pass as `scaler` argument.
                    baseline_lstm, counter_lstm = simulate_policy_lstm_template(features, fire_reduction=fire_red/100.0, traffic_reduction=traffic_red/100.0)
                    st.success(f"LSTM estimate: {baseline_lstm:.0f} → {counter_lstm:.0f} (Δ = {baseline_lstm-counter_lstm:.0f})")
                except Exception as e:
                    st.error(f"Error running LSTM template: {e}")

# ======= END: CONTINUATION CODE =======
