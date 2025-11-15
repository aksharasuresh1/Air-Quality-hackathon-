# app.py
# Streamlit app with robust SMS Alert system + AQI fetchers + policy simulator
# NOTE: Edit constants or environment variables at top as needed.

import os
import json
import time
from datetime import datetime, timedelta
from urllib.parse import urlencode

import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# Optional - Twilio for SMS
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except Exception:
    TWILIO_AVAILABLE = False

# Optional - Tensorflow LSTM template (not used for SMS logic by default)
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

# ---------------------------
# CONFIG / ENV VARS
# ---------------------------
# Twilio credentials (if you want to use Twilio to send SMS)
TWILIO_SID = os.getenv("TWILIO_SID", "")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN", "")
TWILIO_FROM = os.getenv("TWILIO_FROM", "")  # +1xxx
ALERT_HISTORY_FILE = "alerts_sent.json"    # persisted sent alert history (local file)

# FIRMS bounding box for Punjab/Haryana area (adjustable)
FIRMS_MINLAT = 27.0
FIRMS_MINLON = 74.0
FIRMS_MAXLAT = 31.5
FIRMS_MAXLON = 78.5

# ---------------------------
# Utility functions
# ---------------------------
def load_alert_history(path=ALERT_HISTORY_FILE):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {}
    except Exception:
        return {}

def save_alert_history(history, path=ALERT_HISTORY_FILE):
    try:
        with open(path, "w") as f:
            json.dump(history, f, indent=2, default=str)
    except Exception as e:
        st.error(f"Could not save alert history: {e}")

def now_iso():
    return datetime.utcnow().isoformat()

# ---------------------------
# AQI category function (fixed colors; hazardous updated)
# ---------------------------
def get_aqi_category(aqi):
    """Categorizes AQI value and provides color, emoji, and health advice."""
    try:
        aqi = float(aqi)
    except Exception:
        return "Unknown", [128, 128, 128], "❓", "No data"

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
        # HAZARDOUS - changed color to dark maroon to avoid duplication with purple
        return "Hazardous", [102, 0, 51], "☠️", "Health warnings of emergency conditions. The entire population is more likely to be affected."

# ---------------------------
# Simple PM2.5 -> AQI converter (for station PM2.5 proxy if needed)
# ---------------------------
def pm25_to_aqi(pm25_val):
    try:
        c = float(pm25_val)
    except Exception:
        return None

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
    return 500

# ---------------------------
# OpenAQ: fetch latest data (station-wise)
# ---------------------------
@st.cache_data(ttl=300)
def fetch_openaq_latest(city="Delhi", limit=1000):
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
            # fallback to first measurement value if pm25 missing
            if pm25 is None and res.get("measurements"):
                pm25 = res["measurements"][0].get("value")
                last_updated = res["measurements"][0].get("lastUpdated")
            if lat is None or lon is None or pm25 is None:
                continue
            aqi_est = pm25_to_aqi(pm25)
            cat, col, emoji, advice = get_aqi_category(aqi_est)
            rows.append({
                "station_name": location,
                "lat": lat,
                "lon": lon,
                "pm25": pm25,
                "aqi": aqi_est,
                "category": cat,
                "color": col,
                "emoji": emoji,
                "advice": advice,
                "last_updated": last_updated
            })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df['aqi'] = pd.to_numeric(df['aqi'], errors='coerce')
        return df
    except Exception as e:
        st.warning(f"OpenAQ fetch failed: {e}")
        return pd.DataFrame()

# ---------------------------
# NASA FIRMS: count recent fire hotspots in bbox
# ---------------------------
@st.cache_data(ttl=900)
def fetch_firms_count(minlat=FIRMS_MINLAT, minlon=FIRMS_MINLON, maxlat=FIRMS_MAXLAT, maxlon=FIRMS_MAXLON):
    # Use VIIRS 24h CSV (simple approach)
    try:
        csv_url = "https://firms.modaps.eosdis.nasa.gov/viirs/viirs_snpp_24h.csv"
        resp = requests.get(csv_url, timeout=15)
        resp.raise_for_status()
        text = resp.text
        df = pd.read_csv(pd.compat.StringIO(text))
        df_region = df[(df['latitude'] >= minlat) & (df['latitude'] <= maxlat) &
                       (df['longitude'] >= minlon) & (df['longitude'] <= maxlon)]
        return int(len(df_region))
    except Exception as e:
        # don't fail the app on FIRMS issues
        st.info(f"FIRMS fetch error (ignored): {e}")
        return None

# ---------------------------
# Weather fetch (Open-Meteo free endpoint)
# ---------------------------
@st.cache_data(ttl=600)
def fetch_weather_point(lat=28.7041, lon=77.1025):
    try:
        # Using Open-Meteo free endpoint (no API key)
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        payload = r.json()
        return payload.get("current_weather", {})
    except Exception:
        return {}

# ---------------------------
# Aggregation helpers for alerting
# ---------------------------
def aggregate_aqi_stats(df):
    """Return mean_aqi, median_aqi, max_aqi, stations_exceeding dict."""
    if df is None or df.empty:
        return {
            "mean_aqi": None,
            "median_aqi": None,
            "max_aqi": None,
            "stations_exceeding": {},
            "n_stations": 0
        }
    mean_aqi = float(df['aqi'].mean())
    median_aqi = float(df['aqi'].median())
    max_aqi = float(df['aqi'].max())
    # count by category threshold groups
    stations_exceeding = {
        "gt_100": int((df['aqi'] > 100).sum()),
        "gt_200": int((df['aqi'] > 200).sum()),
        "gt_300": int((df['aqi'] > 300).sum())
    }
    return {
        "mean_aqi": mean_aqi,
        "median_aqi": median_aqi,
        "max_aqi": max_aqi,
        "stations_exceeding": stations_exceeding,
        "n_stations": len(df)
    }

# ---------------------------
# SMS send helper with dedupe/hysteresis
# ---------------------------
def send_sms_via_twilio(to_number, body, dry_run=True):
    """Sends SMS via Twilio if configured. If dry_run=True, do not actually send (just log)."""
    if dry_run or not TWILIO_AVAILABLE or not (TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM):
        st.info(f"[DRY-RUN] SMS to {to_number}: {body}")
        return {"status": "dry-run", "to": to_number, "body": body}

    try:
        client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
        msg = client.messages.create(body=body, from_=TWILIO_FROM, to=to_number)
        return {"status": "sent", "sid": msg.sid}
    except Exception as e:
        st.error(f"Twilio send error: {e}")
        return {"status": "error", "error": str(e)}

def should_send_alert(history, area_key, severity_key, min_minutes_between=120):
    """
    Determine if we should send a new alert for (area, severity) based on alert history.
    - history: dict loaded from file
    - area_key: string e.g., 'delhi_zone'
    - severity_key: e.g., 'hazardous'
    - min_minutes_between: minimum cooldown period before sending another alert of same severity
    """
    now = datetime.utcnow()
    area_hist = history.get(area_key, {})
    sev_hist = area_hist.get(severity_key)
    if not sev_hist:
        return True
    try:
        last_ts = datetime.fromisoformat(sev_hist["last_sent"])
        if (now - last_ts) >= timedelta(minutes=min_minutes_between):
            return True
        else:
            return False
    except Exception:
        return True

def register_sent_alert(history, area_key, severity_key, details):
    now = datetime.utcnow().isoformat()
    if area_key not in history:
        history[area_key] = {}
    history[area_key][severity_key] = {"last_sent": now, "details": details}
    return history

# ---------------------------
# Compose SMS text
# ---------------------------
def compose_alert_text(area_name, severity_label, mean_aqi, max_aqi, stations_count):
    # Short, actionable template
    time_str = datetime.now().strftime("%d %b %Y %H:%M")
    advice = {
        "Hazardous": "Avoid all outdoor activity. Wear an N95 mask if outside. Seek medical help if symptoms worsen.",
        "Very Unhealthy": "Reduce outdoor exertion; sensitive groups avoid outdoor activities.",
        "Unhealthy": "Reduce prolonged or heavy outdoor exertion.",
        "Unhealthy for Sensitive Groups": "Sensitive groups reduce prolonged exertion.",
        "Moderate": "Unusually sensitive people should consider reducing exertion.",
        "Good": "Air quality is good."
    }
    message = (f"{area_name} Air Alert ({time_str}): {severity_label}.\n"
               f"Mean AQI: {mean_aqi:.0f}, Peak AQI: {max_aqi:.0f} across {stations_count} stations.\n"
               f"{advice.get(severity_label, '')}")
    return message

# ---------------------------
# Core alert check logic
# ---------------------------
def run_alert_check(aqi_df, area_key="delhi_ncr", area_name="Delhi-NCR", config=None):
    """
    Checks current AQI stats and sends SMS alerts if thresholds exceeded using config.
    config is a dict with:
      - baseline_window_mins: int (rolling window minutes, used if historical series available)
      - min_stations_for_alert: int (how many stations must exceed severity threshold)
      - severity_thresholds: dict mapping severity->AQI entry threshold
      - hysteresis_exit_offsets: dict mapping severity->how much lower AQI must fall to clear
      - cooldown_minutes: int (min minutes between re-sends of same severity)
      - recipients: list of phone numbers
      - dry_run: bool
    """
    config = config or {}
    history = load_alert_history()

    stats = aggregate_aqi_stats(aqi_df)
    mean_aqi = stats['mean_aqi'] if stats['mean_aqi'] is not None else 0
    max_aqi = stats['max_aqi'] if stats['max_aqi'] is not None else 0
    stations_exceeding = stats['stations_exceeding']
    n_stations = stats['n_stations']

    # Determine severity based on max_aqi (or mean if you prefer)
    severity = None
    if max_aqi > config['severity_thresholds']['Hazardous']:
        severity = "Hazardous"
    elif max_aqi > config['severity_thresholds']['Very Unhealthy']:
        severity = "Very Unhealthy"
    elif max_aqi > config['severity_thresholds']['Unhealthy']:
        severity = "Unhealthy"
    elif max_aqi > config['severity_thresholds']['Unhealthy for Sensitive Groups']:
        severity = "Unhealthy for Sensitive Groups"
    elif max_aqi > config['severity_thresholds']['Moderate']:
        severity = "Moderate"
    else:
        severity = "Good"

    # Additional checks: ensure at least X stations exceed the threshold for that severity
    min_stations = config.get('min_stations_for_alert', 3)
    severity_station_count_ok = False
    if severity == "Hazardous":
        severity_station_count_ok = (stations_exceeding.get("gt_300", 0) >= min_stations)
    elif severity == "Very Unhealthy":
        severity_station_count_ok = (stations_exceeding.get("gt_200", 0) >= min_stations)
    elif severity == "Unhealthy":
        severity_station_count_ok = (stations_exceeding.get("gt_150", 0) >= min_stations) if 'gt_150' in stations_exceeding else ((aqi_df['aqi']>150).sum() >= min_stations)
    elif severity == "Unhealthy for Sensitive Groups":
        severity_station_count_ok = (stations_exceeding.get("gt_100", 0) >= min_stations)
    else:
        severity_station_count_ok = False  # no alert for Good/Moderate by default

    # Hysteresis: check last_sent and decide if new alert permitted
    allowed_to_send = should_send_alert(history, area_key, severity, min_minutes_between=config.get('cooldown_minutes', 120))

    # Compose and send
    results = {"should_send": False, "reason": "", "sent": []}

    if severity in ["Hazardous", "Very Unhealthy", "Unhealthy", "Unhealthy for Sensitive Groups"] and severity_station_count_ok:
        if allowed_to_send:
            text = compose_alert_text(area_name, severity, mean_aqi, max_aqi, n_stations)
            recipients = config.get('recipients', [])
            dry_run = config.get('dry_run', True)
            for to in recipients:
                # Basic validation of phone string
                if not to or len(to.strip()) < 6:
                    continue
                send_result = send_sms_via_twilio(to, text, dry_run=dry_run)
                results['sent'].append({"to": to, "result": send_result})
            # register in history
            details = {"severity": severity, "mean": mean_aqi, "max": max_aqi, "n_stations": n_stations}
            history = register_sent_alert(history, area_key, severity, details)
            save_alert_history(history)
            results['should_send'] = True
            results['reason'] = f"{severity} condition met and sent to recipients."
        else:
            results['reason'] = f"{severity} condition met but cooldown prevents sending."
    else:
        results['reason'] = f"No alert criteria matched (severity={severity}, station_count_ok={severity_station_count_ok})."

    return results, stats

# ---------------------------
# Linear policy simulator (simple)
# ---------------------------
def simulate_policy_linear(features, fire_reduction=0.0, traffic_reduction=0.0, industry_reduction=0.0):
    coeffs = {"fire": 1.8, "traffic": 1.5, "industry": 0.6, "wind": -4.0, "intercept": 40.0}
    fire = features['fire_count']
    traffic = features['traffic_index']
    industry = features['industry_index']
    wind = features['wind_speed'] if features.get('wind_speed') is not None else 2.0

    baseline = coeffs['intercept'] + coeffs['fire']*fire + coeffs['traffic']*traffic + coeffs['industry']*industry + coeffs['wind']*wind

    fire_new = fire * (1 - float(fire_reduction))
    traffic_new = traffic * (1 - float(traffic_reduction))
    industry_new = industry * (1 - float(industry_reduction))

    counter = coeffs['intercept'] + coeffs['fire']*fire_new + coeffs['traffic']*traffic_new + coeffs['industry']*industry_new + coeffs['wind']*wind

    return max(0, baseline), max(0, counter), {
        "coeffs": coeffs,
        "inputs": {"fire": fire, "traffic": traffic, "industry": industry, "wind": wind},
        "inputs_new": {"fire": fire_new, "traffic": traffic_new, "industry": industry_new}
    }

# ---------------------------
# Prepare features helper
# ---------------------------
def prepare_current_features(df):
    avg_aqi = float(df['aqi'].mean()) if not df.empty else np.nan
    worst_aqi = float(df['aqi'].max()) if not df.empty else np.nan
    fire_count = fetch_firms_count() or 0
    hour = datetime.now().hour
    traffic_index = 120 if (7 <= hour <= 10 or 17 <= hour <= 20) else 80
    industry_index = 50
    weather = fetch_weather_point()
    wind_speed = weather.get('windspeed') if isinstance(weather, dict) else None
    if wind_speed is None:
        # try alternate naming Open-Meteo returns windspeed
        wind_speed = weather.get('windspeed', 2.0)
    return {
        "avg_aqi": avg_aqi,
        "worst_aqi": worst_aqi,
        "fire_count": fire_count,
        "traffic_index": traffic_index,
        "industry_index": industry_index,
        "wind_speed": float(wind_speed or 2.0)
    }

# ---------------------------
# UI: Streamlit layout
# ---------------------------
st.set_page_config(layout="wide", page_title="AQI Dashboard + Robust SMS Alerts")

st.sidebar.header("Alert System Settings")
dry_run = st.sidebar.checkbox("Dry-run (no real SMS)", value=True)
recipients_raw = st.sidebar.text_area("Recipients (one per line, E.164 or local)", value="+911234567890\n+919876543210", help="Enter phone numbers to send alerts to.")
recipients = [r.strip() for r in recipients_raw.splitlines() if r.strip()]

min_stations_for_alert = st.sidebar.number_input("Min stations to require for alert", min_value=1, max_value=20, value=3)
cooldown_minutes = st.sidebar.number_input("Cooldown between same alerts (minutes)", min_value=15, max_value=1440, value=180)

# thresholds (you can tweak)
st.sidebar.markdown("**Severity thresholds (AQI)**")
s_hazardous = st.sidebar.number_input("Hazardous > ", value=300)
s_very_unhealthy = st.sidebar.number_input("Very Unhealthy > ", value=200)
s_unhealthy = st.sidebar.number_input("Unhealthy > ", value=150)
s_unhealthy_sensitive = st.sidebar.number_input("Unhealthy for Sensitive Groups > ", value=100)
s_moderate = st.sidebar.number_input("Moderate > ", value=50)

config = {
    "severity_thresholds": {
        "Hazardous": s_hazardous,
        "Very Unhealthy": s_very_unhealthy,
        "Unhealthy": s_unhealthy,
        "Unhealthy for Sensitive Groups": s_unhealthy_sensitive,
        "Moderate": s_moderate
    },
    "min_stations_for_alert": int(min_stations_for_alert),
    "cooldown_minutes": int(cooldown_minutes),
    "recipients": recipients,
    "dry_run": dry_run
}

st.title("Interactive AQI Dashboard + Robust SMS Alerts")
st.markdown("Realtime station info sourced from OpenAQ (mirrors local station feeds).")

# Fetch data and show map/table
aqi_df = fetch_openaq_latest(city="Delhi")
if aqi_df.empty:
    st.warning("No station data available from OpenAQ right now.")
else:
    # Map visualization
    fig_map = px.scatter_mapbox(
        aqi_df,
        lat="lat",
        lon="lon",
        color=aqi_df['aqi'],
        size=np.clip(aqi_df['aqi'] / 50, 6, 30),
        hover_name="station_name",
        hover_data=["aqi", "pm25", "last_updated"],
        color_continuous_scale="Turbo",
        zoom=8,
        height=450
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)

    # Station table
    st.dataframe(aqi_df[['station_name', 'aqi', 'pm25', 'category', 'last_updated']].sort_values("aqi", ascending=False).reset_index(drop=True), height=300)

# Policy simulator tab
st.markdown("---")
st.header("Policy Simulator (Quick & Interpretable)")
features = prepare_current_features(aqi_df)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Avg AQI (current)", f"{features['avg_aqi']:.1f}" if not np.isnan(features['avg_aqi']) else "N/A")
    st.write("Fire hotspots (24h):", features['fire_count'])
with col2:
    st.metric("Traffic index (proxy)", f"{features['traffic_index']}")
    st.metric("Industry index (proxy)", f"{features['industry_index']}")
with col3:
    st.metric("Wind speed (m/s)", f"{features['wind_speed']:.1f}")

st.write("#### Adjust policy sliders (demo model uses a simple linear mapping)")
fire_red = st.slider("Reduce fires (stubble burning) %", 0, 100, 20)
traffic_red = st.slider("Reduce traffic emissions %", 0, 100, 40)
industry_red = st.slider("Reduce industry emissions %", 0, 100, 10)

baseline, counterfactual, details = simulate_policy_linear(features, fire_reduction=fire_red/100.0, traffic_reduction=traffic_red/100.0, industry_reduction=industry_red/100.0)
st.write(f"Estimated AQI: **{baseline:.0f}** → **{counterfactual:.0f}**   (Δ = {baseline-counterfactual:.0f})")
fig = px.bar(x=["Baseline","Counterfactual"], y=[baseline,counterfactual], labels={"x":"","y":"Estimated AQI"})
st.plotly_chart(fig, use_container_width=True)

# Alert check UI
st.markdown("---")
st.header("Run SMS Alert Check Now")
st.write("This will evaluate current station-wide AQI and (if criteria met) send alerts to configured recipients.")
if st.button("Run Alert Check Now"):
    result, stats = run_alert_check(aqi_df, area_key="delhi_ncr", area_name="Delhi-NCR", config=config)
    st.write("Result:", result)
    st.write("Stats:", stats)
    st.success("Alert check completed (see result above).")

# Show alert history and allow manual clear
st.markdown("---")
st.header("Alert History (recent)")
history = load_alert_history()
st.write(history)
if st.button("Clear alert history (careful)"):
    save_alert_history({}, ALERT_HISTORY_FILE)
    st.success("Alert history cleared.")

st.markdown("### Notes & tuning tips")
st.markdown("""
- Increase **Min stations** to avoid noisy single-station spikes.
- Use **Cooldown** to prevent repeated identical alerts.
- Use **Dry-run** while testing so SMS are not sent.
- For production SMS: set environment variables TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM and uncheck Dry-run.
- To improve accuracy further, replace the simple linear model with a trained multivariate model (LSTM/XGBoost) and feed the counterfactual scenario into it.
""")
