import streamlit as st
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
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
    
    /* Glassmorphism card styling */
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
    
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .input-section {
        margin-bottom: 1.5rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid rgba(0, 0, 0, 0.08);
    }
    
    .input-section:last-child {
        border-bottom: none;
        margin-bottom: 2rem;
        padding-bottom: 0;
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
    
    .stSlider {
        margin-bottom: 1.5rem;
    }
    
    .stSelectbox {
        margin-bottom: 1.5rem;
    }
    
    .stRadio {
        margin-bottom: 1.5rem;
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
            # Tenure Section
            st.markdown('<div class="input-label">⏱️ Tenure (months)</div>', unsafe_allow_html=True)
            tenure = st.slider(
                'Tenure',
                min_value=0,
                max_value=72,
                value=24,
                step=1,
                label_visibility='collapsed'
            )
            
            # Monthly Charges Section
            st.markdown('<div class="input-label">💳 Monthly Charges ($)</div>', unsafe_allow_html=True)
            monthly_charges = st.slider(
                'Monthly Charges',
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                step=1.0,
                label_visibility='collapsed'
            )
        
        with col2:
            # Contract Type Section
            st.markdown('<div class="input-label">📋 Contract Type</div>', unsafe_allow_html=True)
            contract = st.selectbox(
                'Contract',
                ['Month-to-month', 'One year', 'Two year'],
                label_visibility='collapsed'
            )
            
            # Internet Service Section
            st.markdown('<div class="input-label">🌐 Internet Service</div>', unsafe_allow_html=True)
            internet_service = st.selectbox(
                'Internet Service',
                ['DSL', 'Fiber optic', 'No'],
                label_visibility='collapsed'
            )
        
        # Tech Support Section
        st.markdown('<div class="input-label">🛠️ Tech Support</div>', unsafe_allow_html=True)
        tech_support = st.radio(
            'Tech Support',
            ['Yes', 'No'],
            horizontal=True,
            label_visibility='collapsed'
        )
        
        # Predict Button
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        submit = st.form_submit_button('🚀 Predict Churn', use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'contract': contract,
        'internet_service': internet_service,
        'tech_support': tech_support,
        'submit': submit
    }

def preprocess_input(user_input: Dict) -> pd.DataFrame:
    """Preprocess user input to match model requirements."""
    # Create minimal dataframe with required columns
    input_data = pd.DataFrame([{
        'tenure': user_input['tenure'],
        'MonthlyCharges': user_input['monthly_charges'],
        'Contract': user_input['contract'],
        'InternetService': user_input['internet_service'],
        'TechSupport': user_input['tech_support'],
        # Add default values for required categorical columns
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
    
    # Tenure-based insights
    if user_input['tenure'] < 12:
        insights.append(f"📍 New customer ({user_input['tenure']} months) - Higher churn risk. Invest in onboarding.")
    elif user_input['tenure'] > 48:
        insights.append(f"⭐ Loyal customer ({user_input['tenure']} months) - Consider loyalty rewards.")
    
    # Contract-based insights
    if user_input['contract'] == 'Month-to-month':
        insights.append("🔴 Month-to-month contracts have highest churn. Recommend annual plans.")
    elif user_input['contract'] == 'Two year':
        insights.append("🟢 Long-term contract = lower churn risk. Great for retention.")
    
    # Charges-based insights
    if user_input['monthly_charges'] > 100:
        insights.append(f"💰 High charges (${user_input['monthly_charges']:.0f}/mo) - Verify value delivery.")
    
    # Tech Support insights
    if user_input['tech_support'] == 'No':
        insights.append("🛠️ No tech support - Add support to reduce friction & churn.")
    else:
        insights.append("✅ Tech support active - Positive retention factor.")
    
    # Internet Service insights
    if user_input['internet_service'] == 'Fiber optic':
        insights.append("🚀 Fiber optic service - Premium offering, monitor satisfaction.")
    elif user_input['internet_service'] == 'No':
        insights.append("⚠️ No internet service - Limited engagement opportunity.")
    
    return insights

# ============================================================================
# MAIN APP
# ============================================================================

# Header
st.markdown("""
<div class="header-card">
    <h1 class="header-title">Customer Churn Intelligence</h1>
    <p class="header-subtitle">AI-powered retention insights for your business</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📖 About This Dashboard")
    st.info("""
    **Customer Churn Intelligence** predicts customer churn risk using machine learning.
    
    Identify at-risk customers and take proactive measures to improve retention.
    """)
    
    st.markdown("### 🤖 Model Information")
    st.json({
        "Model Type": "Logistic Regression",
        "Accuracy": "82.11%",
        "Features": "5 Key Inputs",
        "Threshold": "0.65 (High Risk)"
    })
    
    st.markdown("### ❓ How to Use")
    st.markdown("""
    1. **Fill in customer profile** with 5 key metrics
    2. **Click "🚀 Predict Churn"** button
    3. **View results** on the results page
    4. **Take action** based on recommendations
    """)

# Main content - centered layout with input form only
col1, col2, col3 = st.columns([0.5, 2, 0.5])

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    user_input = get_user_input()
    st.markdown('</div>', unsafe_allow_html=True)
    
    if user_input['submit']:
        with st.spinner('🔮 Analyzing customer data...'):
            input_data = preprocess_input(user_input)
            prediction, probabilities = predict_churn(input_data)
            insights = generate_insights(user_input, probabilities[1])
        
        # Store results in session state and navigate to results page
        st.session_state.prediction_result = {
            'prediction': prediction,
            'probabilities': probabilities,
            'insights': insights,
            'user_input': user_input
        }
        
        st.session_state.page = 'results'
        st.rerun()

# Show results if on results page
if st.session_state.get('page') == 'results' and 'prediction_result' in st.session_state:
    result = st.session_state.prediction_result
    prediction = result['prediction']
    probabilities = result['probabilities']
    insights = result['insights']
    user_input = result['user_input']
    
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
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Key Insights
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 💡 Key Insights")
    for insight in insights:
        st.markdown(f'<div class="insight-item">{insight}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Customer Profile Summary
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 👤 Customer Profile Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Tenure", f"{user_input['tenure']} months")
    
    with col2:
        st.metric("Monthly Charges", f"${user_input['monthly_charges']:.2f}")
    
    with col3:
        st.metric("Contract Type", user_input['contract'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Internet Service", user_input['internet_service'])
    
    with col2:
        st.metric("Tech Support", user_input['tech_support'])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Action Buttons
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button('← Back to Prediction'):
            st.session_state.page = 'home'
            st.rerun()
    
    with col2:
        if st.button('🔄 New Prediction'):
            st.session_state.clear()
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
