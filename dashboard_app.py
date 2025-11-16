import streamlit as st
import pandas as pd
import plotly.express as px
import os
import tempfile
from PIL import Image, ImageDraw
from inference_sdk import InferenceHTTPClient

# ======================================================
# 🔹 ROBFLOW YOLO DETECTION SECTION
# ======================================================

st.sidebar.header("🎯 Helmet / No-Helmet / Overloading Detection")

ROBOFLOW_API_KEY = "9hoQyKuZbdUuWxbLIW0r"

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

MODEL_ID = "nohelmet-dnqyh/1"


# ======================================================
# ⚠ UPDATED VIOLATION LOGIC (Same rules as yolo_detector.py)
# ======================================================
def get_violation_only(preds):
    classes = [p["class"] for p in preds]

    has_helmet = "helmet" in classes
    has_no_helmet = "no_helmet" in classes
    has_overloading = "overloading" in classes

    # All three → return both violations
    if has_helmet and has_no_helmet and has_overloading:
        return "🚨 OVERLOADING and 🚨 NO HELMET"

    # no_helmet + overloading → return no_helmet
    if has_no_helmet and has_overloading:
        return "🚨 NO HELMET"

    # only no_helmet
    if has_no_helmet:
        return "🚨 NO HELMET"

    # only overloading
    if has_overloading:
        return "🚨 OVERLOADING"

    # helmet alone or nothing
    return "✅ No Violation Detected"


# ======================================================
# IMAGE UPLOAD + YOLO INFERENCE
# ======================================================

uploaded_file = st.sidebar.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.sidebar.button("Run Detection"):
        # save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            temp_path = tmp.name

        with st.spinner("Running YOLO model..."):
            result = CLIENT.infer(temp_path, model_id=MODEL_ID)

        preds = result.get("predictions", [])
        final_status = get_violation_only(preds)

        # Draw ONLY violation boxes
        draw = ImageDraw.Draw(img)
        for pred in preds:
            if pred["class"] == "helmet":
                continue

            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            left = x - w / 2
            top = y - h / 2
            right = x + w / 2
            bottom = y + h / 2

            draw.rectangle([left, top, right, bottom], outline="red", width=3)
            draw.text((left, top), f"{pred['class']} ({pred['confidence']:.2f})", fill="red")

        # Show final status
        st.subheader("🚦 Violation Result")
        if final_status.startswith("🚨"):
            st.error(final_status)
        else:
            st.success(final_status)

        st.subheader("📸 Detection Result")
        st.image(img, caption="Detected Violations", use_container_width=True)

        st.subheader("📄 Prediction JSON")
        st.json(result)


# ======================================================
# 🔹 PARQUET DATA LOADING
# ======================================================

DATA_DIR = "data/processed/parquet/"

try:
    available_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")]
except FileNotFoundError:
    st.error("❌ Directory missing: data/processed/parquet/")
    st.stop()

st.sidebar.markdown("### 📁 Available Datasets")
for f in available_files:
    st.sidebar.write("✔", f)

@st.cache_data
def safe_load_parquet(filename):
    try:
        return pd.read_parquet(os.path.join(DATA_DIR, filename))
    except:
        return pd.DataFrame()

if "cleaned_violations.parquet" in available_files:
    df = safe_load_parquet("cleaned_violations.parquet")
else:
    st.error("❌ cleaned_violations.parquet not found.")
    st.stop()


# ======================================================
# 🔹 DASHBOARD START
# ======================================================

st.title("🚦 Smart Traffic Violation Analytics Dashboard")
st.write("YOLO Violation Detection + Analytical Reporting Dashboard")

if df.empty:
    st.error("❌ No data found.")
    st.stop()

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")


# ======================================================
# 🔹 FILTERS
# ======================================================

st.sidebar.header("🔍 Filters")

def safe_unique(series):
    return series.dropna().unique().tolist() if series is not None else []


violation_cols = [c for c in df.columns if "violation" in c.lower() and "type" in c.lower()]
violation_col = violation_cols[0] if violation_cols else None

if violation_col:
    violation_options = safe_unique(df[violation_col])
    selected_violation = st.sidebar.multiselect(
        "Violation Type",
        violation_options,
        default=violation_options
    )
else:
    selected_violation = []


severity_cols = [c for c in df.columns if "severity" in c.lower()]
severity_col = severity_cols[0] if severity_cols else None

if severity_col:
    severity_options = safe_unique(df[severity_col])
    selected_severity = st.sidebar.multiselect(
        "Severity",
        severity_options,
        default=severity_options
    )
else:
    selected_severity = []


if "date" in df.columns:
    min_date, max_date = df["date"].min(), df["date"].max()
    date_range = st.sidebar.date_input("Date Range", [min_date, max_date])


# APPLY FILTERS
filtered_df = df.copy()

if violation_col:
    filtered_df = filtered_df[filtered_df[violation_col].isin(selected_violation)]

if severity_col:
    filtered_df = filtered_df[filtered_df[severity_col].isin(selected_severity)]

if "date" in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df["date"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))
    ]


# ======================================================
# 🔹 SHOW FILTERED DATA
# ======================================================

st.subheader("📊 Filtered Data Preview")
st.dataframe(filtered_df.head(20))


# ======================================================
# 🔹 TIME TRENDS
# ======================================================

st.subheader("📅 Time Trends")

if "date" in filtered_df.columns:
    daily = filtered_df.groupby(filtered_df["date"].dt.date).size().reset_index(name="count")
    st.plotly_chart(px.line(daily, x="date", y="count",
                            title="Violations Per Day", markers=True))

if "hour" not in filtered_df.columns and "date" in filtered_df.columns:
    filtered_df["hour"] = filtered_df["date"].dt.hour

if "hour" in filtered_df.columns:
    hourly = filtered_df.groupby("hour").size().reset_index(name="count")
    st.plotly_chart(px.bar(hourly, x="hour", y="count",
                           title="Violations Per Hour"))


# ======================================================
# 🔹 VIOLATION TYPE DISTRIBUTION
# ======================================================

if violation_col:
    st.subheader("🚨 Violation Type Distribution")
    counts = filtered_df[violation_col].value_counts().reset_index()
    counts.columns = [violation_col, "count"]
    st.plotly_chart(px.bar(counts, x=violation_col, y="count"))


# ======================================================
# 🔹 HIGH-RISK LOCATIONS
# ======================================================

st.subheader("📍 High-Risk Locations")

loc_cols = [c for c in df.columns if "location" in c.lower()]
if loc_cols:
    top_loc = filtered_df[loc_cols[0]].value_counts().head(10).reset_index()
    top_loc.columns = ["Location", "Count"]
    st.table(top_loc)


# ======================================================
# 🔹 EXPORT RESULTS
# ======================================================

st.subheader("📤 Export Results")

os.makedirs("reports", exist_ok=True)

if st.button("Export CSV"):
    filtered_df.to_csv("reports/filtered_summary.csv", index=False)
    st.success("Saved: reports/filtered_summary.csv")

if st.button("Export JSON"):
    filtered_df.to_json("reports/filtered_summary.json", orient="records", indent=2)
    st.success("Saved: reports/filtered_summary.json")


# ======================================================
# FOOTER
# ======================================================

st.markdown("---")
st.caption("Developed by **Shivani Marelli** | Smart Traffic Project – YOLO + Analytics Dashboard")
