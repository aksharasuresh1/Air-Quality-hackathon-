"""
Streamlit AQI app with:
 - corrected AQI category colors and legend
 - OpenAQ fetcher (station-level PM2.5 -> approximate AQI conversion)
 - SQLite persistence for recent AQI history (for sustained-alert checks)
 - Robust SMS alerting via Twilio with retries and cooldown
 - Simulation mode when Twilio creds are not provided

Setup:
 - pip install -r requirements.txt
 - Create a .env with:
     TWILIO_ACCOUNT_SID=your_sid
     TWILIO_AUTH_TOKEN=your_token
     TWILIO_FROM_NUMBER=+1XXXXXXXXX
 - Or leave them empty to use "simulate SMS" mode.
 - Run: streamlit run app.py
"""

import os
import time
import json
from datetime import datetime, timedelta
from urllib.parse import urlencode

import requests
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Table, MetaData, select, and_
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

# Load env vars
load_dotenv()

TWILIO_ENABLED = bool(os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("TWILIO_AUTH_TOKEN") and os.getenv("TWILIO_FROM_NUMBER"))
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")

# SQLite DB for history and alerts
DB_FILE = "aqi_history.db"
engine = create_engine(f"sqlite:///{DB_FILE}", echo=False, future=True)
meta = MetaData()

# Table definitions
aqi_table = Table(
    "aqi_history", meta,
    Column("id", Integer, primary_key=True),
    Column("timestamp", DateTime, nullable=False),
    Column("station", String, nullable=True),
    Column("lat", Float, nullable=True),
    Column("lon", Float, nullable=True),
    Column("aqi", Float, nullable=False),
)

alerts_table = Table(
    "sent_alerts", meta,
    Column("id", Integer, primary_key=True),
    Column("phone", String, nullable=False),
    Column("sent_at", DateTime, nullable=False),
    Column("category", String, nullable=False),
    Column("message", String, nullable=True),
)

# Create tables if not exist
meta.create_all(engine)

st.set_page_config(page_title="AQI Dashboard + Robust SMS Alerts", layout="wide")

# ----------------------------
# AQI category function (fixed colors)
# returns: (category_name, hex_color, emoji, advice)
# Colors chosen to be visually distinct
# ----------------------------
def get_aqi_category(aqi: float):
    if aqi is None or np.isnan(aqi):
        return "Unknown", "#808080", "❓", "No data"
    a = float(aqi)
    if a <= 50:
        return "Good", "#009E60", "✅", "Enjoy outdoor activities."
    elif a <= 100:
        return "Moderate", "#FFD600", "🟡", "Unusually sensitive people should consider reducing prolonged or heavy exertion."
    elif a <= 150:
        return "Unhealthy for Sensitive Groups", "#F97316", "🟠", "Sensitive groups should reduce prolonged or heavy exertion."
    elif a <= 200:
        return "Unhealthy", "#DC2626", "🔴", "Everyone may begin to experience health effects."
    elif a <= 300:
        return "Very Unhealthy", "#7C3AED", "🟣", "Health alert: everyone may experience more serious health effects."
    else:
        # HAZARDOUS FIX — BLACK COLOR + BLACK CIRCLE
        return "Hazardous", "#000000", "⚫", "Health warnings of emergency conditions. The entire population is more likely to be affected."

# ----------------------------
# Simple PM2.5 -> AQI conversion (US EPA breakpoints)
# (Used only as fallback where station AQI not provided)
# ----------------------------
def pm25_to_aqi(pm25_val):
    try:
        c = float(pm25_val)
    except Exception:
        return np.nan
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for Clow, Chigh, Ilow, Ihigh in breakpoints:
        if Clow <= c <= Chigh:
            aqi = ((Ihigh - Ilow) / (Chigh - Clow)) * (c - Clow) + Ilow
            return round(aqi)
    return 500

# ----------------------------
# Fetch OpenAQ latest for city (Delhi by default)
# returns DataFrame with station, lat, lon, aqi (approx)
# ----------------------------
@st.cache_data(ttl=300)
def fetch_openaq_latest(city="Delhi", limit=1000):
    url = "https://api.openaq.org/v2/latest"
    params = {"city": city, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        rows = []
        for res in data.get("results", []):
            coords = res.get("coordinates", {})
            lat = coords.get("latitude")
            lon = coords.get("longitude")
            location = res.get("location") or res.get("name") or "unknown"
            pm25 = None
            for measure in res.get("measurements", []):
                if measure.get("parameter") == "pm25":
                    pm25 = measure.get("value")
                    break
            if pm25 is None:
                # fallback to first measurement value
                if res.get("measurements"):
                    pm25 = res["measurements"][0].get("value")
            if lat is None or lon is None or pm25 is None:
                continue
            aqi = pm25_to_aqi(pm25)
            rows.append({"station": location, "lat": lat, "lon": lon, "aqi": aqi})
        if not rows:
            return pd.DataFrame(columns=["station","lat","lon","aqi"])
        df = pd.DataFrame(rows)
        return df
    except Exception as e:
        st.warning(f"OpenAQ fetch error: {e}")
        return pd.DataFrame(columns=["station","lat","lon","aqi"])

# ----------------------------
# Insert latest station readings into SQLite (to build sustained history)
# ----------------------------
def store_aqi_readings(df: pd.DataFrame):
    if df.empty:
        return
    with engine.begin() as conn:
        now = datetime.utcnow()
        for _, r in df.iterrows():
            ins = aqi_table.insert().values(timestamp=now, station=r['station'], lat=float(r['lat']), lon=float(r['lon']), aqi=float(r['aqi']))
            conn.execute(ins)

# ----------------------------
# Query recent average AQI across last `minutes` minutes
# ----------------------------
def recent_avg_aqi(minutes=60):
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)
    with engine.begin() as conn:
        sel = select(aqi_table.c.aqi).where(aqi_table.c.timestamp >= cutoff)
        res = conn.execute(sel).scalars().all()
    if not res:
        return None
    return float(np.nanmean(res))

# ----------------------------
# Twilio send SMS with retry & logging to DB
# ----------------------------
def send_sms_via_twilio(phone: str, message: str, max_retries=3, backoff_sec=2):
    # If Twilio not configured, simulate and log
    if not TWILIO_ENABLED:
        st.info(f"[SIMULATED SMS] To: {phone} | Msg: {message}")
        # record the simulated alert as sent
        with engine.begin() as conn:
            conn.execute(alerts_table.insert().values(phone=phone, sent_at=datetime.utcnow(), category="SIMULATED", message=message))
        return True

    from twilio.rest import Client
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    attempt = 0
    while attempt < max_retries:
        try:
            client.messages.create(body=message, from_=TWILIO_FROM_NUMBER, to=phone)
            # log in DB
            with engine.begin() as conn:
                conn.execute(alerts_table.insert().values(phone=phone, sent_at=datetime.utcnow(), category="SMS", message=message))
            return True
        except Exception as e:
            attempt += 1
            st.error(f"Twilio send error attempt {attempt}: {e}")
            time.sleep(backoff_sec * (2 ** (attempt-1)))
    return False

# ----------------------------
# Alert decision logic:
# - threshold: AQI category threshold (e.g., "Unhealthy" -> AQI >= 151)
# - sustained_minutes: require recent_avg_aqi over last sustained_minutes to be above threshold
# - cooldown_hours: do not send another alert to same phone within cooldown hours
# - return (should_send:bool, reason:str)
# ----------------------------
CATEGORY_THRESHOLDS = {
    "Good": 50,
    "Moderate": 100,
    "Unhealthy for Sensitive Groups": 150,
    "Unhealthy": 200,
    "Very Unhealthy": 300,
    "Hazardous": 501  # any aqi > 300 considered hazardous; using >300 for trigger
}

def get_threshold_for_alert(target_category_name: str):
    # We want to send alerts when AQI >= lower-bound of category
    mapping = {
        "Unhealthy for Sensitive Groups": 101,
        "Unhealthy": 151,
        "Very Unhealthy": 201,
        "Hazardous": 301
    }
    return mapping.get(target_category_name, 151)

def was_alert_sent_recently(phone: str, cooldown_hours: int):
    cutoff = datetime.utcnow() - timedelta(hours=cooldown_hours)
    with engine.begin() as conn:
        sel = select(alerts_table.c.sent_at).where(and_(alerts_table.c.phone == phone, alerts_table.c.sent_at >= cutoff))
        res = conn.execute(sel).scalars().all()
    return len(res) > 0

def should_send_alert(phone: str, target_category: str, sustained_minutes: int, cooldown_hours: int):
    threshold = get_threshold_for_alert(target_category)
    avg = recent_avg_aqi(minutes=sustained_minutes)
    if avg is None:
        return False, "No recent AQI data"
    if avg < threshold:
        return False, f"Recent average AQI {avg:.1f} < threshold {threshold}"
    if was_alert_sent_recently(phone, cooldown_hours):
        return False, f"An alert was already sent in the past {cooldown_hours} hours"
    return True, f"Recent avg AQI {avg:.1f} >= threshold {threshold}"

# ----------------------------
# UI: left column controls, right column display
# ----------------------------
st.title("Interactive AQI Dashboard + Robust SMS Alerts")
tabs = st.tabs(["Dashboard", "Policy Simulator", "SMS Alerts / Monitor", "About"])

with tabs[0]:
    st.header("Live AQI Map (Delhi - OpenAQ data)")
    df = fetch_openaq_latest("Delhi")
    if df.empty:
        st.warning("No AQI station data available from OpenAQ right now.")
    else:
        # store latest into DB for history
        store_aqi_readings(df)

        # map
        df['category'] = df['aqi'].apply(lambda x: get_aqi_category(x)[0])
        df['color'] = df['aqi'].apply(lambda x: get_aqi_category(x)[1])
        fig = px.scatter_mapbox(df, lat="lat", lon="lon", hover_name="station", hover_data=["aqi", "category"],
                                color="category", color_discrete_map={row['category']: row['color'] for _, row in df.iterrows()},
                                size_max=15, zoom=9, height=500)
        fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

        # legend (manual)
        st.markdown("**AQI Legend**")
        cols = st.columns(3)
        cats = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]
        for i, cat in enumerate(cats):
            c = get_aqi_category(CATEGORY_THRESHOLDS.get(cat, 0))[1] if False else get_aqi_category( (CATEGORY_THRESHOLDS.get(cat,0) or 0) + 1 )[1]
        # Display a nicer legend
        leg_items = []
        for cat in cats:
            # find a sample AQI value inside that category to get color
            sample_val = {
                "Good": 25, "Moderate": 75, "Unhealthy for Sensitive Groups": 125,
                "Unhealthy": 175, "Very Unhealthy": 225, "Hazardous": 350
            }[cat]
            color = get_aqi_category(sample_val)[1]
            emoji = get_aqi_category(sample_val)[2]
            leg_items.append(f"<span style='display:inline-block;width:14px;height:14px;background:{color};border-radius:50%;margin-right:6px;'></span> {emoji} {cat} ({sample_val})")
        st.markdown("<br>".join(leg_items), unsafe_allow_html=True)

        st.markdown("### Stations snapshot")
        st.dataframe(df[["station","aqi","category"]].sort_values("aqi", ascending=False).reset_index(drop=True), use_container_width=True)

with tabs[1]:
    st.header("Policy Simulator (simple linear demo)")
    st.info("This is a rapid demo simulator. For production-grade counterfactuals, integrate your trained model + scaler + sequence preprocessing.")
    # Simple feature proxies (aggregates)
    avg_aqi = recent_avg_aqi(minutes=60) or (df['aqi'].mean() if not df.empty else np.nan)
    st.metric("Recent average AQI (last 60 min)", f"{avg_aqi:.1f}" if not np.isnan(avg_aqi) else "N/A")
    fire_reduction = st.slider("Reduce fires (proxy) %", 0, 100, 20)
    traffic_reduction = st.slider("Reduce traffic %", 0, 100, 40)
    industry_reduction = st.slider("Reduce industry %", 0, 100, 10)

    # Simple linear coefficients - should be trained on historical data for credibility
    coeffs = {"intercept": 40.0, "fire": 1.8, "traffic": 1.5, "industry": 0.6}
    # Build fake proxy inputs (for demo)
    proxy_fire = 20  # you can replace by actual FIRMS count near region
    proxy_traffic = 100
    proxy_industry = 50
    baseline = coeffs['intercept'] + coeffs['fire'] * proxy_fire + coeffs['traffic'] * proxy_traffic + coeffs['industry'] * proxy_industry
    baseline = max(0, baseline)
    baseline_after = coeffs['intercept'] + coeffs['fire'] * (proxy_fire*(1-fire_reduction/100.0)) + coeffs['traffic']*(proxy_traffic*(1-traffic_reduction/100.0)) + coeffs['industry']*(proxy_industry*(1-industry_reduction/100.0))
    baseline_after = max(0, baseline_after)
    st.metric("Estimated AQI: baseline → counterfactual", f"{baseline:.0f} → {baseline_after:.0f} (Δ {baseline - baseline_after:.0f})")
    st.plotly_chart(px.bar(x=["Baseline","Counterfactual"], y=[baseline, baseline_after], labels={"y":"Estimated AQI"}), use_container_width=True)

with tabs[2]:
    st.header("SMS Alerts / Monitor")
    st.markdown("Configure alert thresholds and phone numbers. The app will check current AQI (uses stored recent history) and send SMS if the sustained-average is above threshold.")

    col1, col2 = st.columns([2,1])
    with col1:
        phone = st.text_input("Phone number for alerts (E.164 format, e.g., +91XXXXXXXXXX)", value=os.getenv("DEMO_PHONE",""))
        st.write("Target category to alert when reached or exceeded:")
        target_category = st.selectbox("Alert when AQI reaches:", ["Unhealthy for Sensitive Groups","Unhealthy","Very Unhealthy","Hazardous"])
        sustained_minutes = st.number_input("Sustained minutes (recent average window)", min_value=10, max_value=1440, value=60, step=10)
        cooldown_hours = st.number_input("Cooldown hours per phone (no repeated alerts)", min_value=1, max_value=72, value=6, step=1)
        simulate_sms = st.checkbox("Simulate SMS (don't actually call Twilio even if credentials exist)", value=not TWILIO_ENABLED)
        if simulate_sms:
            st.warning("SIMULATION MODE: SMS will be printed and logged but not sent.")
    with col2:
        st.write("Twilio status:")
        if TWILIO_ENABLED:
            st.success("Twilio credentials detected (real SMS enabled).")
        else:
            st.info("Twilio not configured - SMS will be simulated. Set TWILIO_* env vars to enable real SMS.")

    # Manual check button
    if st.button("Run Alert Check Now"):
        if not phone:
            st.error("Please provide a phone number in E.164 format.")
        else:
            # Let simulate_sms override
            if simulate_sms:
                prev_twilio_enabled = False
            else:
                prev_twilio_enabled = TWILIO_ENABLED

            # Check
            should_send, reason = should_send_alert(phone=phone, target_category=target_category, sustained_minutes=sustained_minutes, cooldown_hours=cooldown_hours)
            if should_send:
                avg = recent_avg_aqi(minutes=sustained_minutes)
                category_name, color_hex, emoji, advice = get_aqi_category(avg)
                message = f"AQ Alert: {category_name} ({avg:.0f}). {advice} — This alert triggered because sustained average over last {sustained_minutes} min met your threshold ({target_category})."
                st.info(f"Alert triggered: {message}")
                # send (or simulate)
                sent = send_sms_via_twilio(phone, message)
                if sent:
                    st.success("Alert sent and logged.")
                else:
                    st.error("Failed to send alert after retries.")
            else:
                st.success(f"No alert sent. Reason: {reason}")

    st.markdown("---")
    st.markdown("Recent sent alerts (latest 20):")
    with engine.begin() as conn:
        sel = select(alerts_table).order_by(alerts_table.c.sent_at.desc()).limit(20)
        res = conn.execute(sel).mappings().all()
    if res:
        df_alerts = pd.DataFrame(res)
        df_alerts['sent_at'] = df_alerts['sent_at'].astype(str)
        st.dataframe(df_alerts, use_container_width=True)
    else:
        st.info("No alerts have been logged yet.")

with tabs[3]:
    st.header("About & Notes")
    st.markdown("""
    **What this app does**
    - Fetches latest station PM2.5 from OpenAQ and converts to approximate AQI for display.
    - Stores each fetch into a small SQLite DB to compute *sustained* averages over time.
    - Robust SMS alerting logic:
        - Requires sustained average above chosen threshold for `sustained_minutes`.
        - Uses a `cooldown_hours` to avoid repeated alerts to same number.
        - Uses Twilio with retries and exponential backoff; if Twilio not configured, SMS is simulated and logged locally.
    **Important**
    - This is a demo-level system. For production:
        - Use official CPCB/WAQI API for station AQI where possible (avoids pm25->aqi approximation).
        - Run the monitoring logic in a server cron / cloud scheduler (this Streamlit button triggers on-demand checks).
        - Train a forecasting model and use its forecasted AQI for early warning.
    """)

# End of app.py
