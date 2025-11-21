import streamlit as st
import pandas as pd
import plotly.express as px
import os
from PIL import Image, ImageDraw
from ultralytics import YOLO

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Smart Traffic Violation Dashboard",
    page_icon="🚦",
    layout="wide"
)

# ======================================================
# CUSTOM CSS
# ======================================================
st.markdown("""
<style>

[data-testid="stSidebar"] {
    background-color: #B2BEB5;
    padding: 20px;
}
[data-testid="stSidebar"] * {
    color: #000000 !important;
    font-size: 15px;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e0e0e0;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    text-align: center;
}
.card h2 {
    font-size: 30px;
    color: #333333;
}
.card p {
    font-size: 16px;
    color: #777777;
    margin-top: -10px;
}

.section-box {
    background: white;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #e0e0e0;
    box-shadow: 0 3px 15px rgba(0,0,0,0.1);
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================
st.markdown("""
<h1 style='text-align: center; color:#222;'>
🚦 SMART TRAFFIC VIOLATION PATTERN DETECTOR
</h1>
""", unsafe_allow_html=True)

# ======================================================
# LOAD DATA
# ======================================================
DATA_DIR = "data/processed/parquet/"

@st.cache_data
def load_parquet(name):
    try:
        return pd.read_parquet(DATA_DIR + name)
    except:
        return pd.DataFrame()

df = load_parquet("cleaned_violations.parquet")
violations_by_window = load_parquet("violations_by_window.parquet")
weekday_weekend_analysis = load_parquet("weekday_weekend_analysis.parquet")
violation_time_patterns = load_parquet("violation_time_patterns.parquet")

if df.empty:
    st.error("❌ cleaned_violations.parquet missing")
    st.stop()

# AUTO-DETECT COLUMN NAMES
def find_col(keys):
    for key in keys:
        for col in df.columns:
            if key.lower() in col.lower().replace(" ", ""):
                return col
    return None

timestamp_col = find_col(["timestamp", "time"])
violation_col = find_col(["violationtype", "violation"])
severity_col = find_col(["severity"])
location_col = find_col(["location", "loc"])

# ======================================================
# LOAD YOLO MODEL
# ======================================================
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train2/weights/last.pt")

model = load_model()

# ======================================================
# SIDEBAR FILTERS
# ======================================================
st.sidebar.header("🔍 Filters")

with st.sidebar.expander("📁 Available Datasets", expanded=False):
    for f in os.listdir(DATA_DIR):
        st.write("✔", f)

def unique(series): 
    return sorted(series.dropna().unique().tolist())

selected_violation = st.sidebar.multiselect(
    "Violation Type", unique(df[violation_col]), default=unique(df[violation_col])
)

selected_severity = st.sidebar.multiselect(
    "Severity", unique(df[severity_col]), default=unique(df[severity_col])
)

# APPLY FILTERS
filtered_df = df.copy()
filtered_df = filtered_df[filtered_df[violation_col].isin(selected_violation)]
filtered_df = filtered_df[filtered_df[severity_col].isin(selected_severity)]

# ======================================================
# KPI CARDS
# ======================================================
total_viol = len(filtered_df)
common_violation = filtered_df[violation_col].mode()[0]

filtered_df["hour"] = pd.to_datetime(filtered_df[timestamp_col], errors="coerce").dt.hour
peak_hour = int(filtered_df["hour"].mode()[0])

c1, c2, c3 = st.columns(3)
c1.markdown(f"<div class='card'><h2>{total_viol}</h2><p>Total Violations</p></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='card'><h2>{common_violation}</h2><p>Most Common Violation</p></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='card'><h2>{peak_hour}:00</h2><p>Peak Hour</p></div>", unsafe_allow_html=True)

# ======================================================
# YOLO DETECTION (ABOVE FILTERED DATA PREVIEW)
# ======================================================
st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.subheader("🖼️ YOLO Violation Detection")

upload = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if upload:
    image = Image.open(upload).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Detecting..."):
        results = model(image, save=False)

    r = results[0]
    boxes = r.boxes

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    violations = []

    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cls = model.names[int(b.cls[0])]
        conf = float(b.conf[0])

        violations.append(cls)

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 12), f"{cls} ({conf:.2f})", fill="red")

    st.markdown("### 📌 Result:")
    if violations:
        st.success("Violations Detected:")
        for v in list(dict.fromkeys(violations)):
            st.write(f"- **{v}**")
    else:
        st.warning("📱 USING_MOBILE — No violations detected")

    st.image(annotated, caption="YOLO Output", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# FILTERED DATA (Appears BELOW YOLO)
# ======================================================
st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.subheader("📊 Filtered Data Preview")
st.dataframe(filtered_df.head(20))
st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIOLATION TYPE DISTRIBUTION
# ======================================================
st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.subheader("🚨 Violation Type Distribution")
viol_count = filtered_df[violation_col].value_counts().reset_index()
viol_count.columns = [violation_col, "Count"]
st.plotly_chart(px.bar(viol_count, x=violation_col, y="Count", color="Count"), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# TIME WINDOW CHART
# ======================================================
if not violations_by_window.empty:
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.subheader("⏳ Violations per 3-Hour Window")
    st.plotly_chart(px.bar(violations_by_window, x="TimeWindow", y="Total_Violations", color="Total_Violations"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# WEEKDAY VS WEEKEND
# ======================================================
if not weekday_weekend_analysis.empty:
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.subheader("🗓️ Weekday vs Weekend Violations")
    weekday_weekend_analysis["DayType"] = weekday_weekend_analysis["IsWeekend"].map({True: "Weekend", False: "Weekday"})
    st.plotly_chart(px.bar(weekday_weekend_analysis, x="DayType", y="Total_Violations", color="Violation Type"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# HEATMAP
# ======================================================
if not violation_time_patterns.empty:
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.subheader("🔥 Violation Types vs Time Window")
    pivot = violation_time_patterns.pivot_table(index="Violation Type", columns="TimeWindow", values="Violations", fill_value=0)
    st.plotly_chart(px.imshow(pivot, color_continuous_scale="Blues"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# HOTSPOT MAP (Corrected)
# ======================================================
st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.subheader("📍 Hotspot Map")

if location_col:
    loc = filtered_df[location_col].value_counts().reset_index()
    loc.columns = ["Location", "Count"]

    def split_pos(x):
        try:
            la, lo = x.split(",")
            return float(la), float(lo)
        except:
            return None, None

    loc["lat"], loc["lon"] = zip(*loc["Location"].apply(split_pos))
    map_df = loc.dropna()

    # 🔥 Filter only valid INDIA coordinates
    map_df = map_df[
        (map_df["lat"] >= 8) & (map_df["lat"] <= 37) &
        (map_df["lon"] >= 68) & (map_df["lon"] <= 97)
    ]

    if not map_df.empty:
        fig = px.scatter_mapbox(
            map_df,
            lat="lat",
            lon="lon",
            size="Count",
            hover_name="Location",
            zoom=4
        )
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No valid India coordinates found.")
else:
    st.info("Location column missing.")


# ======================================================
# EXPORT SECTION
# ======================================================
st.subheader("📤 Export Results")

os.makedirs("reports", exist_ok=True)

if st.button("Export CSV"):
    filtered_df.to_csv("reports/filtered_summary.csv", index=False)
    st.success("CSV saved!")

if st.button("Export JSON"):
    filtered_df.to_json("reports/filtered_summary.json", orient="records", indent=2)
    st.success("JSON saved!")

st.markdown("---")
st.caption("Developed by **Shivani Marelli** | Smart Traffic Project")
