import streamlit as st
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title='Prediction Results',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    body {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
    }
    
    .card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.25);
        margin-bottom: 1.5rem;
    }
    
    .header-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        padding: 3rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.95;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    .result-badge {
        display: inline-block;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 1rem 0;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        color: white;
    }
    
    .metric-card {
        background: rgba(102, 126, 234, 0.08);
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .insight-item {
        padding: 1rem;
        background: rgba(102, 126, 234, 0.05);
        border-radius: 8px;
        margin-bottom: 0.75rem;
        border-left: 3px solid #667eea;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 3rem;
        border-radius: 12px;
        font-size: 1.05rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        cursor: pointer;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CHECK IF RESULTS EXIST IN SESSION STATE
# ============================================================================
if 'prediction_result' not in st.session_state:
    st.warning('⚠️ No prediction results found. Please go back and make a prediction.')
    if st.button('← Go Back to Prediction'):
        st.switch_page("app.py")
    st.stop()

# Get results from session state
prediction = st.session_state.prediction_result['prediction']
probabilities = st.session_state.prediction_result['probabilities']
insights = st.session_state.prediction_result['insights']
user_input = st.session_state.prediction_result['user_input']

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="header-card">
    <h1 class="header-title">🎯 Prediction Results</h1>
    <p class="header-subtitle">AI-powered churn prediction analysis</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN RESULTS
# ============================================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 🎯 Prediction Results")

churn_prob = probabilities[1]

# Determine risk level
if churn_prob >= 0.65:
    risk_level = "🔴 High Risk"
    risk_class = "risk-high"
    recommendation = "⚠️ Immediate action required. Offer retention incentive."
elif churn_prob >= 0.35:
    risk_level = "🟡 Moderate Risk"
    risk_class = "risk-low"
    recommendation = "📋 Review engagement strategy. Consider proactive outreach."
else:
    risk_level = "🟢 Low Risk"
    risk_class = "risk-low"
    recommendation = "✅ Customer is satisfied. Maintain current service level."

# Display risk badge
st.markdown(f'<div class="result-badge {risk_class}">{risk_level}</div>', unsafe_allow_html=True)

# Display probability metrics in columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.85rem; color: #667; margin-bottom: 0.5rem;">Churn Probability</div>
        <div style="font-size: 2rem; font-weight: 700; color: #667eea;">{churn_prob:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.85rem; color: #667; margin-bottom: 0.5rem;">Retention Probability</div>
        <div style="font-size: 2rem; font-weight: 700; color: #51cf66;">{(1-churn_prob):.1%}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.85rem; color: #667; margin-bottom: 0.5rem;">Prediction</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: {'#ff6b6b' if prediction == 1 else '#51cf66'};">
            {'CHURN' if prediction == 1 else 'RETAIN'}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Progress bar visualization
st.markdown("### Churn Probability Gauge")
fig = go.Figure(data=[go.Bar(
    x=[churn_prob],
    orientation='h',
    marker=dict(
        color=f'rgba({int(255 * churn_prob)}, {int(51)}, {int(107)}, 0.8)',
        line=dict(color='rgba(102, 126, 234, 0.3)', width=2)
    ),
    text=f'{churn_prob:.1%}',
    textposition='auto',
    hovertemplate='%{x:.1%}<extra></extra>'
)])
fig.update_layout(
    xaxis=dict(range=[0, 1], tickformat='.0%'),
    height=100,
    margin=dict(l=0, r=0, t=0, b=0),
    showlegend=False,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
st.plotly_chart(fig, use_container_width=True)

# Recommendation
st.markdown(f"""
<div style="background: rgba(102, 126, 234, 0.1); border-left: 4px solid #667eea; padding: 1.5rem; border-radius: 8px; margin-top: 1.5rem;">
    <div style="font-weight: 600; color: #333; margin-bottom: 0.5rem;">💡 Recommendation</div>
    <div style="color: #555;">{recommendation}</div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# KEY INSIGHTS
# ============================================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 💡 Key Insights")
for insight in insights:
    st.markdown(f'<div class="insight-item">{insight}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# CUSTOMER PROFILE SUMMARY
# ============================================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 👤 Customer Profile Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Tenure", f"{user_input['tenure']} months")

with col2:
    st.metric("Monthly Charges", f"${user_input['monthly_charges']:.2f}")

with col3:
    st.metric("Contract Type", user_input['contract'])

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Internet Service", user_input['internet_service'])

with col2:
    st.metric("Tech Support", user_input['tech_support'])

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# ACTION BUTTONS
# ============================================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    if st.button('← Back to Prediction'):
        st.session_state.clear()
        st.switch_page("app.py")

with col2:
    if st.button('🔄 New Prediction'):
        st.session_state.clear()
        st.switch_page("app.py")

st.markdown('</div>', unsafe_allow_html=True)
