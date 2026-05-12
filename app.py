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
    page_title='Customer Churn Intelligence',
    page_icon='🎯',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS for premium SaaS styling
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
    
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .input-label {
        font-weight: 600;
        color: #555;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    
    .button-container {
        display: flex;
        justify-content: center;
        margin-top: 2rem;
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
        st.error('❌ Model artifacts not found. Run churn_prediction.py first.')
        st.stop()
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

model, ct = load_model_artifacts()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_user_input() -> Dict:
    """Get simplified user input for prediction."""
    with st.form('prediction_form', border=False):
        st.markdown('<div class="section-title">📊 Customer Profile</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="input-label">⏱️ Tenure (months)</div>', unsafe_allow_html=True)
            tenure = st.slider('Tenure', min_value=0, max_value=72, value=24, step=1, label_visibility='collapsed')
            
            st.markdown('<div class="input-label">💳 Monthly Charges ($)</div>', unsafe_allow_html=True)
            monthly_charges = st.slider('Monthly Charges', min_value=0.0, max_value=200.0, value=70.0, step=1.0, label_visibility='collapsed')
        
        with col2:
            st.markdown('<div class="input-label">📋 Contract Type</div>', unsafe_allow_html=True)
            contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'], label_visibility='collapsed')
            
            st.markdown('<div class="input-label">🌐 Internet Service</div>', unsafe_allow_html=True)
            internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'], label_visibility='collapsed')
        
        st.markdown('<div class="input-label">🛠️ Tech Support</div>', unsafe_allow_html=True)
        tech_support = st.radio('Tech Support', ['Yes', 'No'], horizontal=True, label_visibility='collapsed')
        
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        submit = st.form_submit_button('🚀 Predict Churn', use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    return {'tenure': tenure, 'monthly_charges': monthly_charges, 'contract': contract, 'internet_service': internet_service, 'tech_support': tech_support, 'submit': submit}

def preprocess_input(user_input: Dict) -> pd.DataFrame:
    """Preprocess user input to match model requirements."""
    input_data = pd.DataFrame([{
        'tenure': user_input['tenure'],
        'MonthlyCharges': user_input['monthly_charges'],
        'Contract': user_input['contract'],
        'InternetService': user_input['internet_service'],
        'TechSupport': user_input['tech_support'],
        'gender': 'Male',
        'SeniorCitizen': 'No',
        'Partner': 'Yes',
        'Dependents': 'No',
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'PaperlessBilling': 'No',
        'PaymentMethod': 'Electronic check',
        'TotalCharges': user_input['tenure'] * user_input['monthly_charges'],
    }])
    return input_data

def predict_churn(input_data: pd.DataFrame) -> Tuple[int, np.ndarray]:
    """Predict churn using the loaded model."""
    input_encoded = ct.transform(input_data)
    prediction = model.predict(input_encoded)[0]
    probabilities = model.predict_proba(input_encoded)[0]
    return prediction, probabilities

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
    """Create churn vs retention comparison chart based on prediction."""
    churn_prob = probabilities[1]
    retention_prob = probabilities[0]
    
    fig = go.Figure(data=[
        go.Bar(name='Retention', x=['Probability'], y=[retention_prob], marker=dict(color='#51cf66', line=dict(color='rgba(81, 207, 102, 0.5)', width=2)), text=f'{retention_prob:.1%}', textposition='auto'),
        go.Bar(name='Churn', x=['Probability'], y=[churn_prob], marker=dict(color='#ff6b6b', line=dict(color='rgba(255, 107, 107, 0.5)', width=2)), text=f'{churn_prob:.1%}', textposition='auto')
    ])
    
    fig.update_layout(height=300, margin=dict(l=50, r=20, t=20, b=50), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', barmode='stack', yaxis=dict(tickformat='.0%', range=[0, 1]), xaxis_showgrid=False, yaxis_showgrid=True, yaxis_gridcolor='rgba(0,0,0,0.05)', hovermode='x unified', showlegend=True)
    return fig

def create_risk_factor_analysis(user_input: Dict, churn_prob: float) -> go.Figure:
    """Create risk factor analysis based on input features."""
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

def display_results(prediction: int, probabilities: np.ndarray, insights: list):
    """Display prediction results in a visually appealing way."""
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
# MAIN APP
# ============================================================================

st.markdown("""
<div class="header-card">
    <h1 class="header-title">Customer Churn Intelligence</h1>
    <p class="header-subtitle">AI-powered retention insights for your business</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📖 About This Dashboard")
    st.info("""
    **Customer Churn Intelligence** predicts customer churn risk using machine learning.
    
    Identify at-risk customers and take proactive measures to improve retention.
    """)
    
    st.markdown("### 🤖 Model Information")
    st.json({"Model Type": "Logistic Regression", "Accuracy": "82.11%", "Features": "5 Key Inputs", "Threshold": "0.65 (High Risk)"})
    
    st.markdown("### ❓ How to Use")
    st.markdown("""
    1. **Fill in customer profile** with 5 key metrics
    2. **Click "🚀 Predict Churn"** button
    3. **Review results** and insights
    4. **Take action** based on recommendations
    """)

col_input, col_output = st.columns([1, 1.1], gap="medium")

with col_input:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    user_input = get_user_input()
    st.markdown('</div>', unsafe_allow_html=True)

with col_output:
    if user_input['submit']:
        with st.spinner('🔮 Analyzing customer data...'):
            input_data = preprocess_input(user_input)
            prediction, probabilities = predict_churn(input_data)
            insights = generate_insights(user_input, probabilities[1])
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Prediction Results")
        display_results(prediction, probabilities, insights)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 💡 Key Insights")
        for insight in insights:
            st.markdown(f'<div class="insight-item">{insight}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        importance_df = get_feature_importance()
        if importance_df is not None:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📊 Feature Importance")
            fig_imp = go.Figure(data=[go.Bar(y=importance_df['Feature'], x=importance_df['Importance'], orientation='h', marker=dict(color=importance_df['Importance'], colorscale='Purples', line=dict(color='rgba(102, 126, 234, 0.3)', width=1)), hovertemplate='%{y}: %{x:.3f}<extra></extra>')])
            fig_imp.update_layout(height=350, margin=dict(l=150, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_showgrid=True, xaxis_gridwidth=1, xaxis_gridcolor='rgba(0,0,0,0.05)', yaxis_showgrid=False)
            st.plotly_chart(fig_imp, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 Churn vs Retention Probability")
        fig_comparison = create_churn_vs_retention_chart(probabilities)
        st.plotly_chart(fig_comparison, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Risk Factor Analysis")
        st.markdown("*Individual contribution of each factor to churn risk*")
        fig_risk_factors = create_risk_factor_analysis(user_input, probabilities[1])
        st.plotly_chart(fig_risk_factors, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        ### 👋 Welcome!
        
        Fill in the customer profile on the left and click **"🚀 Predict Churn"** to get started.
        
        The AI model will:
        - ✅ Analyze customer characteristics
        - 🎯 Predict churn risk
        - 💡 Generate actionable insights
        - 📊 Show feature importance
        """)
        st.markdown('</div>', unsafe_allow_html=True)
