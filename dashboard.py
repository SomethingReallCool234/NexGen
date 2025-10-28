import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# --------------------------
# PAGE CONFIGURATION
# --------------------------
st.set_page_config(
    page_title="NexGen Logistics Dashboard",
    layout="wide",
)

# --------------------------
# LOAD DATA
# --------------------------
@st.cache_data
def load_data():
    delivery_df = pd.read_csv("dataset/delivery_performance.csv")
    orders_df = pd.read_csv("dataset/orders.csv")
    feedback_df = pd.read_csv("dataset/customer_feedback.csv")
    routes_df = pd.read_csv("dataset/routes_distance.csv")
    costs_df = pd.read_csv("dataset/cost_breakdown.csv")
    return delivery_df, orders_df, feedback_df, routes_df, costs_df

delivery_df, orders_df, feedback_df, routes_df, costs_df = load_data()

# --------------------------
# LOAD MODEL
# --------------------------
@st.cache_resource
def load_model():
    delay_model = joblib.load("models/delay_predictor.pkl")
    return delay_model

delay_model = load_model()

# --------------------------
# HEADER
# --------------------------
st.markdown(
    """
    <h1 style='text-align:center; margin-bottom:30px;'>NexGen Logistics: Delivery Performance Dashboard</h1>
    """,
    unsafe_allow_html=True
)

# --------------------------
# LAYOUT
# --------------------------
charts_col, controls_col = st.columns([4, 1.5], gap="large")

# --------------------------
# FILTERS
# --------------------------
with controls_col:
    st.subheader("🔍 Filters")
    carrier = st.selectbox("Carrier", ["All"] + sorted(delivery_df["Carrier"].dropna().unique().tolist()))
    status = st.selectbox("Delivery Status", ["All"] + sorted(delivery_df["Delivery_Status"].dropna().unique().tolist()))
    min_rating, max_rating = st.slider("Customer Rating Range", 0.0, 5.0, (0.0, 5.0), step=0.5)

# Apply filters
filtered_df = delivery_df.copy()
if carrier != "All":
    filtered_df = filtered_df[filtered_df["Carrier"] == carrier]
if status != "All":
    filtered_df = filtered_df[filtered_df["Delivery_Status"] == status]
filtered_df = filtered_df[
    (filtered_df["Customer_Rating"] >= min_rating) & (filtered_df["Customer_Rating"] <= max_rating)
]

# --------------------------
# DASHBOARD CHARTS
# --------------------------
with charts_col:
    st.subheader("Delivery Status Distribution")
    status_fig = px.histogram(filtered_df, x="Delivery_Status", color="Delivery_Status", title="Delivery Status Overview")
    st.plotly_chart(status_fig, use_container_width=True)

    st.subheader("Quality Issues")
    issues_fig = px.bar(filtered_df, x="Quality_Issue", color="Quality_Issue", title="Reported Quality Issues Count")
    st.plotly_chart(issues_fig, use_container_width=True)

    st.subheader("Customer Ratings")
    ratings_fig = px.box(filtered_df, y="Customer_Rating", color="Delivery_Status", points="all", title="Customer Rating Distribution by Status")
    st.plotly_chart(ratings_fig, use_container_width=True)

# --------------------------
# PREDICTION SECTION
# --------------------------
st.markdown("---")
st.markdown("<h2 style='text-align:center;'>Predictive Delivery Optimizer</h2>", unsafe_allow_html=True)

st.subheader("Predict Delay Risk Before Dispatch")

col1, col2, col3 = st.columns(3)
with col1:
    carrier = st.selectbox("Carrier", sorted(delivery_df["Carrier"].dropna().unique().tolist()), key="plan_carrier")
    priority = st.selectbox("Priority", sorted(orders_df["Priority"].dropna().unique().tolist()), key="plan_priority")
with col2:
    promised_days = st.number_input("Promised Delivery Days", min_value=1, value=3, key="plan_promised")
    route = st.selectbox("Route", sorted(routes_df["Route"].dropna().unique().tolist()), key="plan_route")
with col3:
    weather_impact = st.selectbox("Weather Impact", sorted(routes_df["Weather_Impact"].dropna().unique().tolist()), key="plan_weather")

# Prepare input
input_data = pd.DataFrame([{
    "Carrier": carrier,
    "Priority": priority,
    "Promised_Delivery_Days": promised_days,
    "Route": route,
    "Weather_Impact": weather_impact
}])

# Predict
if st.button("Predict Delay Risk"):
    try:
        prob = delay_model.predict_proba(input_data)[:, 1][0]
        pred = "High Delay Risk" if prob >= 0.5 else "Low Delay Risk"

        st.markdown(f"### Prediction: **{pred}**")
        st.progress(float(prob))
        if prob >= 0.5:
            st.warning(f"High delay probability: {prob:.2%}. Consider alternate route or early dispatch.")
        else:
            st.success(f"Low delay probability: {prob:.2%}. Shipment likely on time.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --------------------------
# DATA TABLE + DOWNLOAD
# --------------------------
st.markdown("---")
st.subheader("Filtered Delivery Records")
st.dataframe(filtered_df, use_container_width=True)

csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Filtered Data", csv, "filtered_data.csv", "text/csv")

st.markdown("<p style='text-align:center; margin-top:40px; color:gray;'>© 2025 NexGen Logistics Dashboard</p>", unsafe_allow_html=True)
