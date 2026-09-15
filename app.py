import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Package Delivery Time Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Logistics Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0369A1 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 16px;
        padding: 26px 32px;
        margin-bottom: 24px;
        box-shadow: 0 12px 28px -6px rgba(0, 0, 0, 0.35);
    }

    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        margin-bottom: 10px;
    }

    .eta-card {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.15) 0%, rgba(2, 132, 199, 0.05) 100%);
        border: 1px solid rgba(56, 189, 248, 0.4);
        border-radius: 16px;
        padding: 26px;
        text-align: center;
    }

    .eta-hero {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1px;
        color: #38BDF8;
        margin: 6px 0;
    }

    .logistics-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #0EA5E9;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin-top: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to find assets
def get_asset_path(filename):
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path(filename),
        script_dir / filename,
        script_dir / "Package_Delivery_Prediction_Streamlit" / filename,
        script_dir.parent / filename,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return filename

@st.cache_resource
def load_model():
    model_path = get_asset_path("package_delivery_model.pkl")
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Header
st.markdown("""
<div class="main-header">
    <div class="badge-pill">Supply Chain & Last-Mile Logistics AI</div>
    <h1 style="color: #F8FAFC; margin: 0; font-weight: 800; font-size: 2.2rem;">📦 Package Delivery Time Predictor</h1>
    <p style="color: #94A3B8; margin-top: 8px; margin-bottom: 0; font-size: 1.05rem;">
        Forecast package arrival duration (hours and days) using machine learning based on distance, parcel weight, traffic conditions, weather, fleet vehicle, and priority tier.
    </p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🚀 Shipment ETA Calculator", "📁 Manifest Batch Screening (CSV)", "📊 Feature Importance & Diagnostics"])

# --- TAB 1: Shipment ETA Calculator ---
with tabs[0]:
    st.subheader("Package Specifications & Route Factors")

    # Quick Presets
    p_cols = st.columns([1, 1, 1, 3])
    with p_cols[0]:
        load_express = st.button("⚡ Express Van Delivery", width="stretch")
    with p_cols[1]:
        load_freight = st.button("🚚 Heavy Freight Long-Haul", width="stretch")
    with p_cols[2]:
        load_bike = st.button("🚲 Local Courier Bike", width="stretch")

    if load_express:
        st.session_state["dist"] = 28.0
        st.session_state["weight"] = 3.5
        st.session_state["traffic"] = "Medium"
        st.session_state["weather"] = "Clear"
        st.session_state["vehicle"] = "Van"
        st.session_state["warehouse"] = 0.5
        st.session_state["handling"] = 0.5
        st.session_state["priority"] = "Express"
    elif load_freight:
        st.session_state["dist"] = 320.0
        st.session_state["weight"] = 65.0
        st.session_state["traffic"] = "High"
        st.session_state["weather"] = "Storm"
        st.session_state["vehicle"] = "Truck"
        st.session_state["warehouse"] = 4.0
        st.session_state["handling"] = 3.5
        st.session_state["priority"] = "Standard"
    elif load_bike:
        st.session_state["dist"] = 6.0
        st.session_state["weight"] = 1.2
        st.session_state["traffic"] = "Low"
        st.session_state["weather"] = "Clear"
        st.session_state["vehicle"] = "Bike"
        st.session_state["warehouse"] = 0.2
        st.session_state["handling"] = 0.5
        st.session_state["priority"] = "Standard"

    c_left, c_right = st.columns(2, gap="large")

    with c_left:
        st.markdown("#### 📍 Route & Parcel Specs")
        distance = st.slider(
            "Transit Distance (km)", 0.5, 500.0,
            value=float(st.session_state.get("dist", 25.0)), step=1.0,
            key="input_dist"
        )
        weight = st.slider(
            "Package Weight (kg)", 0.1, 100.0,
            value=float(st.session_state.get("weight", 4.0)), step=0.5,
            key="input_weight"
        )
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            warehouse = st.slider(
                "Warehouse Delay (Hours)", 0.0, 24.0,
                value=float(st.session_state.get("warehouse", 1.0)), step=0.5,
                key="input_warehouse"
            )
        with col_t2:
            handling = st.slider(
                "Sorting / Handling (Hours)", 0.0, 12.0,
                value=float(st.session_state.get("handling", 1.0)), step=0.5,
                key="input_handling"
            )

    with c_right:
        st.markdown("#### 🌦️ Route Environment & Fleet Service")
        sub1, sub2 = st.columns(2)
        with sub1:
            traffic = st.selectbox(
                "Traffic Density",
                ["Low", "Medium", "High"],
                index=["Low", "Medium", "High"].index(st.session_state.get("traffic", "Medium")),
                key="input_traffic"
            )
        with sub2:
            weather = st.selectbox(
                "Weather Conditions",
                ["Clear", "Rain", "Storm"],
                index=["Clear", "Rain", "Storm"].index(st.session_state.get("weather", "Clear")),
                key="input_weather"
            )

        sub3, sub4 = st.columns(2)
        with sub3:
            vehicle = st.selectbox(
                "Fleet Vehicle",
                ["Bike", "Van", "Truck"],
                index=["Bike", "Van", "Truck"].index(st.session_state.get("vehicle", "Van")),
                key="input_vehicle"
            )
        with sub4:
            priority = st.selectbox(
                "Delivery Priority Tier",
                ["Standard", "Express"],
                index=["Standard", "Express"].index(st.session_state.get("priority", "Standard")),
                key="input_priority"
            )

    st.markdown("---")
    eval_btn = st.button("🚀 Calculate Estimated Delivery Time (ETA)", type="primary", width="stretch")

    shipment_df = pd.DataFrame([{
        "distance_km": distance,
        "package_weight_kg": weight,
        "traffic_level": traffic,
        "weather": weather,
        "vehicle_type": vehicle,
        "warehouse_delay_hours": warehouse,
        "handling_time_hours": handling,
        "priority": priority
    }])

    pred_hours = max(0.2, float(model.predict(shipment_df)[0]))
    pred_days = pred_hours / 24.0

    st.markdown("### 📋 Shipment ETA Outcome")
    r1, r2 = st.columns([1.3, 1.7], gap="medium")

    with r1:
        st.markdown(f"""
        <div class="eta-card">
            <span style="font-size: 2.8rem;">⏱️</span>
            <div style="color: #38BDF8; font-weight: 700; font-size: 1.15rem; text-transform: uppercase; letter-spacing: 1px;">
                Estimated Transit Time
            </div>
            <div class="eta-hero">
                {pred_hours:.1f} <span style="font-size: 1.2rem; color: #94A3B8;">hours</span>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); padding: 10px; border-radius: 10px; margin-top: 10px;">
                <span style="color: #94A3B8; font-size: 0.95rem;">Approximate Duration:</span>
                <strong style="color: #34D399; font-size: 1.25rem;"> {pred_days:.2f} days</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown("#### 🔍 Transit Delays & Dispatch Factors")
        factors = []
        if priority == "Express":
            factors.append(("Express Priority Routing", "Fast-track sorting and direct non-stop dispatch.", "good"))
        if weather == "Storm":
            factors.append(("Severe Weather Caution", "Storm slows transit velocity and creates safety holding periods.", "warn"))
        if traffic == "High":
            factors.append(("Congested Corridor", "High traffic gridlock adds significant travel friction.", "warn"))
        if warehouse + handling >= 4.0:
            factors.append(("Hub Dwell Overhead", f"Combined hub sorting ({warehouse + handling:.1f} hrs) contributes heavily to ETA.", "warn"))

        if factors:
            for title, desc, tone in factors:
                if tone == "good":
                    st.success(f"**{title}**: {desc}")
                else:
                    st.warning(f"**{title}**: {desc}")
        else:
            st.info("Transit factors align with baseline logistics schedule.")

        st.markdown(f"""
        <div class="logistics-box">
            <strong style="color: #38BDF8;">Dispatch Recommendation:</strong><br>
            <span style="color: #CBD5E1; font-size: 0.92rem;">
                Target delivery SLA commitment: <strong>{pred_hours + 1.5:.1f} hours</strong> (includes standard 90-minute courier buffer).
            </span>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔍 View Raw Features"):
        st.dataframe(shipment_df, width="stretch")

# --- TAB 2: Manifest Batch Screening ---
with tabs[1]:
    st.subheader("Batch Shipment Manifest Screening")
    st.write("Upload a courier manifest CSV or evaluate against the baseline delivery dataset.")

    csv_file = st.file_uploader("Upload Manifest CSV", type=["csv"], key="manifest_csv")
    df_manifest = None

    if csv_file is not None:
        df_manifest = pd.read_csv(csv_file)
        st.info(f"Loaded {len(df_manifest)} shipment records from file.")
    else:
        sample_path = get_asset_path("data/package_delivery_data.csv")
        if os.path.exists(sample_path):
            if st.checkbox("Load baseline delivery dataset (`data/package_delivery_data.csv`)", value=True):
                df_manifest = pd.read_csv(sample_path)
                st.info(f"Loaded {len(df_manifest)} records from baseline dataset.")

    if df_manifest is not None:
        req_cols = ["distance_km", "package_weight_kg", "traffic_level", "weather", "vehicle_type", "warehouse_delay_hours", "handling_time_hours", "priority"]
        missing = [c for c in req_cols if c not in df_manifest.columns]
        if missing:
            st.error(f"Missing columns in dataset: {missing}")
        else:
            if st.button("⚡ Run Batch Manifest Valuation", type="primary"):
                with st.spinner("Predicting delivery transit times..."):
                    preds = model.predict(df_manifest[req_cols])
                    preds = np.maximum(0.2, preds)

                    res_df = df_manifest.copy()
                    res_df["Predicted_Delivery_Hours"] = np.round(preds, 2)
                    res_df["Predicted_Delivery_Days"] = np.round(preds / 24.0, 2)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Shipments", len(res_df))
                    m2.metric("Average ETA", f"{np.mean(preds):.1f} hrs")
                    m3.metric("Fastest Delivery", f"{np.min(preds):.1f} hrs")
                    m4.metric("Longest Delivery", f"{np.max(preds):.1f} hrs")

                    f1, f2 = st.columns(2)
                    with f1:
                        v_filter = st.selectbox("Filter Vehicle:", ["All"] + list(df_manifest["vehicle_type"].unique()))
                    with f2:
                        p_filter = st.selectbox("Filter Priority:", ["All"] + list(df_manifest["priority"].unique()))

                    view = res_df
                    if v_filter != "All":
                        view = view[view["vehicle_type"] == v_filter]
                    if p_filter != "All":
                        view = view[view["priority"] == p_filter]

                    st.dataframe(view, width="stretch")

                    csv_export = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Scored Manifest as CSV",
                        data=csv_export,
                        file_name="package_delivery_eta_predictions.csv",
                        mime="text/csv"
                    )

# --- TAB 3: Model Diagnostics ---
with tabs[2]:
    st.subheader("Model Architecture & Feature Importance")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        #### 🤖 Predictive Delivery Architecture
        - **Algorithm**: `RandomForestRegressor(n_estimators=200, random_state=42)`
        - **Pipeline Preprocessing**:
            - `OneHotEncoder` on `traffic_level`, `weather`, `vehicle_type`, `priority`
            - Passthrough on numerical metrics (`distance_km`, `package_weight_kg`, `warehouse_delay_hours`, `handling_time_hours`)
        - **Accuracy Benchmark**:
            - **R² Score**: **0.8501**
            - **MAE (Mean Absolute Error)**: ~1.53 hours
            - **RMSE**: ~1.87 hours
        """)

    with c2:
        img_path = get_asset_path("feature_importance.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Top Factors Influencing Delivery Time", width="stretch")
        else:
            st.info("Feature importance image not found.")

st.caption("Supply Chain & Logistics Fleet Analytics • Scikit-learn & Streamlit")
