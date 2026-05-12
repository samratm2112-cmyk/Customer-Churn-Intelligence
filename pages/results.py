import streamlit as st
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Tuple

# ============================================================================
# PAGE CONFIG & STYLING
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
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-size: 0.95rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        cursor: pointer;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
    }
    
    h1, h2, h3 {
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL & PREPROCESSOR
# ============================================================================
@st.cache_resource
def load_model_artifacts():
    """Load pre-trained model and preprocessor."""
    model_path = Path('best_model.pkl')
    preprocessor_path = Path('column_transformer.pkl')
    
    if not model_path.exists() or not preprocessor_path.exists():
        st.error('❌ Model artifacts not found.')
        st.stop()
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

model, ct = load_model_artifacts()

# ============================================================================
# CHECK RESULTS EXIST
# ============================================================================
if 'prediction_result' not in st.session_state:
    st.warning('⚠️ No prediction results found. Please go back and make a prediction.')
    if st.button('← Go Back'):
        st.switch_page("app")
    st.stop()

result = st.session_state.prediction_result
prediction = result['prediction']
probabilities = result['probabilities']
user_input = result['user_input']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_insights(user_input: Dict, churn_prob: float) -> list:
    """Generate actionable insights based on customer profile."""
    insights = []
    
    if user_input['tenure'] < 12:
        insights.append(f"📍 New customer ({user_input['tenure']} months) - Higher churn risk. Invest in onboarding.")
    elif user_input['tenure'] > 48:
        insights.append(f"⭐ Loyal customer ({user_input['tenure']} months) - Consider loyalty rewards.")
    
    if user_input['contract'] == 'Month-to-month':
        insights.append("🔴 Month-to-month contracts have highest churn. Recommend annual plans.")
    elif user_input['contract'] == 'Two year':
        insights.append("🟢 Long-term contract = lower churn risk. Great for retention.")
    
    if user_input['monthly_charges'] > 100:
        insights.append(f"💰 High charges (${user_input['monthly_charges']:.0f}/mo) - Verify value delivery.")
    
    if user_input['tech_support'] == 'No':
        insights.append("🛠️ No tech support - Add support to reduce friction & churn.")
    else:
        insights.append("✅ Tech support active - Positive retention factor.")
    
    if user_input['internet_service'] == 'Fiber optic':
        insights.append("🚀 Fiber optic service - Premium offering, monitor satisfaction.")
    elif user_input['internet_service'] == 'No':
        insights.append("⚠️ No internet service - Limited engagement opportunity.")
    
    return insights

def get_feature_importance() -> pd.DataFrame:
    """Get feature importance from the model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = ct.get_feature_names_out()
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=True).tail(8)
        return importance_df
    return None

def create_churn_vs_retention_chart(probabilities: np.ndarray) -> go.Figure:
    """Create churn vs retention comparison chart."""
    churn_prob = probabilities[1]
    retention_prob = probabilities[0]
    
    fig = go.Figure(data=[
        go.Bar(name='Retention', x=['Probability'], y=[retention_prob], marker=dict(color='#51cf66', line=dict(color='rgba(81, 207, 102, 0.5)', width=2)), text=f'{retention_prob:.1%}', textposition='auto'),
        go.Bar(name='Churn', x=['Probability'], y=[churn_prob], marker=dict(color='#ff6b6b', line=dict(color='rgba(255, 107, 107, 0.5)', width=2)), text=f'{churn_prob:.1%}', textposition='auto')
    ])
    
    fig.update_layout(height=300, margin=dict(l=50, r=20, t=20, b=50), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', barmode='stack', yaxis=dict(tickformat='.0%', range=[0, 1]), xaxis_showgrid=False, yaxis_showgrid=True, yaxis_gridcolor='rgba(0,0,0,0.05)', hovermode='x unified', showlegend=True)
    return fig

def create_risk_factor_analysis(user_input: Dict, churn_prob: float) -> go.Figure:
    """Create risk factor analysis."""
    factors = []
    scores = []
    colors = []
    
    tenure_factor = 1 - (user_input['tenure'] / 72)
    factors.append(f"Tenure\n({user_input['tenure']}m)")
    scores.append(tenure_factor)
    colors.append('rgba(255, 107, 107, 0.8)' if tenure_factor > 0.5 else 'rgba(81, 207, 102, 0.8)')
    
    charges_factor = user_input['monthly_charges'] / 200
    factors.append(f"Monthly Charges\n(${user_input['monthly_charges']:.0f})")
    scores.append(charges_factor)
    colors.append('rgba(255, 107, 107, 0.8)' if charges_factor > 0.5 else 'rgba(81, 207, 102, 0.8)')
    
    contract_risk = {'Month-to-month': 0.9, 'One year': 0.4, 'Two year': 0.1}
    contract_factor = contract_risk[user_input['contract']]
    factors.append(f"Contract\n({user_input['contract'][:4]})")
    scores.append(contract_factor)
    colors.append('rgba(255, 107, 107, 0.8)' if contract_factor > 0.5 else 'rgba(81, 207, 102, 0.8)')
    
    internet_risk = {'DSL': 0.3, 'Fiber optic': 0.5, 'No': 0.7}
    internet_factor = internet_risk[user_input['internet_service']]
    factors.append(f"Internet\n({user_input['internet_service'][:3]})")
    scores.append(internet_factor)
    colors.append('rgba(255, 107, 107, 0.8)' if internet_factor > 0.5 else 'rgba(81, 207, 102, 0.8)')
    
    tech_factor = 0.2 if user_input['tech_support'] == 'Yes' else 0.8
    factors.append(f"Tech Support\n({user_input['tech_support']})")
    scores.append(tech_factor)
    colors.append('rgba(255, 107, 107, 0.8)' if tech_factor > 0.5 else 'rgba(81, 207, 102, 0.8)')
    
    fig = go.Figure(data=[go.Bar(x=factors, y=scores, marker=dict(color=colors, line=dict(color='rgba(102, 126, 234, 0.3)', width=1)), text=[f'{s:.0%}' for s in scores], textposition='auto', hovertemplate='%{x}<br>Risk Factor: %{y:.0%}<extra></extra>')])
    
    fig.update_layout(height=350, margin=dict(l=50, r=20, t=20, b=80), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_showgrid=False, yaxis_showgrid=True, yaxis_gridcolor='rgba(0,0,0,0.05)', yaxis=dict(tickformat='.0%', range=[0, 1]), xaxis_tickangle=-45)
    return fig

def display_results(prediction: int, probabilities: np.ndarray):
    """Display prediction results."""
    churn_prob = probabilities[1]
    
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
    
    st.markdown(f'<div class="result-badge {risk_class}">{risk_level}</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><div style="font-size: 0.85rem; color: #667; margin-bottom: 0.5rem;">Churn Probability</div><div style="font-size: 2rem; font-weight: 700; color: #667eea;">{churn_prob:.1%}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div style="font-size: 0.85rem; color: #667; margin-bottom: 0.5rem;">Retention Probability</div><div style="font-size: 2rem; font-weight: 700; color: #51cf66;">{(1-churn_prob):.1%}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div style="font-size: 0.85rem; color: #667; margin-bottom: 0.5rem;">Prediction</div><div style="font-size: 1.8rem; font-weight: 700; color: {"#ff6b6b" if prediction == 1 else "#51cf66"}>{"CHURN" if prediction == 1 else "RETAIN"}</div></div>', unsafe_allow_html=True)
    
    st.markdown("### Churn Probability Gauge")
    fig = go.Figure(data=[go.Bar(x=[churn_prob], orientation='h', marker=dict(color=f'rgba({int(255 * churn_prob)}, {int(51)}, {int(107)}, 0.8)', line=dict(color='rgba(102, 126, 234, 0.3)', width=2)), text=f'{churn_prob:.1%}', textposition='auto', hovertemplate='%{x:.1%}<extra></extra>')])
    fig.update_layout(xaxis=dict(range=[0, 1], tickformat='.0%'), height=100, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f'<div style="background: rgba(102, 126, 234, 0.1); border-left: 4px solid #667eea; padding: 1.5rem; border-radius: 8px; margin-top: 1.5rem;"><div style="font-weight: 600; color: #333; margin-bottom: 0.5rem;">💡 Recommendation</div><div style="color: #555;">{recommendation}</div></div>', unsafe_allow_html=True)

# ============================================================================
# MAIN PAGE
# ============================================================================

st.markdown("""
<div class="header-card">
    <h1 class="header-title">🎯 Prediction Results</h1>
    <p class="header-subtitle">AI-powered churn prediction analysis</p>
</div>
""", unsafe_allow_html=True)

# Results Section
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 🎯 Prediction Results")
display_results(prediction, probabilities)
st.markdown('</div>', unsafe_allow_html=True)

# Key Insights
insights = generate_insights(user_input, probabilities[1])
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 💡 Key Insights")
for insight in insights:
    st.markdown(f'<div class="insight-item">{insight}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Feature Importance
importance_df = get_feature_importance()
if importance_df is not None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Feature Importance")
    fig_imp = go.Figure(data=[go.Bar(y=importance_df['Feature'], x=importance_df['Importance'], orientation='h', marker=dict(color=importance_df['Importance'], colorscale='Purples', line=dict(color='rgba(102, 126, 234, 0.3)', width=1)), hovertemplate='%{y}: %{x:.3f}<extra></extra>')])
    fig_imp.update_layout(height=350, margin=dict(l=150, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_showgrid=True, xaxis_gridwidth=1, xaxis_gridcolor='rgba(0,0,0,0.05)', yaxis_showgrid=False)
    st.plotly_chart(fig_imp, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Churn vs Retention Chart
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📊 Churn vs Retention Probability")
fig_comparison = create_churn_vs_retention_chart(probabilities)
st.plotly_chart(fig_comparison, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Risk Factor Analysis
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 🔍 Risk Factor Analysis")
st.markdown("*Individual contribution of each factor to churn risk*")
fig_risk_factors = create_risk_factor_analysis(user_input, probabilities[1])
st.plotly_chart(fig_risk_factors, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Navigation Buttons
st.markdown('<div class="card">', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    if st.button('← Back to Prediction'):
        st.session_state.clear()
        st.switch_page("app")

with col2:
    if st.button('🔄 New Prediction'):
        st.session_state.clear()
        st.switch_page("app")

st.markdown('</div>', unsafe_allow_html=True)
