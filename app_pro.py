import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# -------- FORCE DARK THEME CSS --------
st.markdown("""
<style>
html, body, [class*="css"]  {
    background-color: #0f172a !important;
    color: white !important;
}

/* Glass card */
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    margin-bottom: 20px;
}

/* Header */
.header {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    margin-bottom: 10px;
}

.sub {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 20px;
}

/* Result styles */
.high {
    border-left: 6px solid red;
}
.low {
    border-left: 6px solid lime;
}
</style>
""", unsafe_allow_html=True)

# -------- HEADER --------
st.markdown('<div class="header">🚀 Customer Churn Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">AI-powered retention insights</div>', unsafe_allow_html=True)

# -------- LAYOUT --------
col1, col2 = st.columns([1, 1.7])

# -------- INPUT --------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Customer Profile")

    tenure = st.slider("Tenure", 0, 72, 12)
    monthly = st.slider("Monthly Charges", 0, 200, 70)

    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet", ["Fiber optic", "DSL", "None"])
    tech = st.radio("Tech Support", ["Yes", "No"])

    btn = st.button("🚀 Predict Churn")

    st.markdown('</div>', unsafe_allow_html=True)

# -------- PREDICTION --------
def predict():
    score = 0
    if tenure < 12: score += 2
    if monthly > 80: score += 2
    if contract == "Month-to-month": score += 2
    if tech == "No": score += 1
    if internet == "Fiber optic": score += 1
    return min(score/8, 1)

# -------- OUTPUT --------
with col2:
    if btn:
        prob = predict()
        churn = prob > 0.5

        cls = "high" if churn else "low"
        label = "HIGH RISK" if churn else "LOW RISK"
        color = "red" if churn else "lime"

        st.markdown(f"""
        <div class="card {cls}" style="text-align:center;">
            <h1 style="color:{color};">{label}</h1>
            <h2>{prob*100:.1f}% probability</h2>
        </div>
        """, unsafe_allow_html=True)

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            gauge={'axis': {'range': [0,100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Insights
        st.markdown("### 💡 Insights")

        if monthly > 80:
            st.warning("High charges increase churn")
        if tenure < 12:
            st.warning("New customers churn more")
        if contract == "Month-to-month":
            st.warning("Flexible contracts churn more")
        if tech == "Yes":
            st.success("Tech support improves retention")

    else:
        st.markdown('<div class="card">👉 Enter inputs and predict</div>', unsafe_allow_html=True)