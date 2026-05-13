import streamlit as st
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from typing import Dict

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
    3. **View results** on the next page
    4. **Take action** based on recommendations
    """)

col1, col2, col3 = st.columns([0.5, 2, 0.5])

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    user_input = get_user_input()
    st.markdown('</div>', unsafe_allow_html=True)
    
    if user_input['submit']:
        with st.spinner('🔮 Analyzing customer data...'):
            try:
                input_data = preprocess_input(user_input)
                # Transform data to numpy array (no column name validation needed)
                input_data_transformed = ct.transform(input_data)
                
                # If it's a sparse matrix, convert to dense array
                if hasattr(input_data_transformed, 'toarray'):
                    input_data_transformed = input_data_transformed.toarray()
                
                # Make predictions with numpy array directly
                prediction = model.predict(input_data_transformed)[0]
                probabilities = model.predict_proba(input_data_transformed)[0]
                
                # Store in session state
                st.session_state.prediction_result = {
                    'prediction': prediction,
                    'probabilities': probabilities,
                    'user_input': user_input
                }
                
                # Navigate to results page
                st.switch_page("pages/results")
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.info("💡 Tip: Please check your input values and try again.")
                import traceback
                with st.expander("Debug Info"):
                    st.code(traceback.format_exc())
