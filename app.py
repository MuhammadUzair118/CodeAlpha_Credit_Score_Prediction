import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==============================================================================
# 1. PREMIUM BRANDED SYSTEM ARCHITECTURE
# ==============================================================================
st.set_page_config(
    page_title="AuraRisk // Credit Scoring OS",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom World-Class Dark Minimalist UI Layout via Injection
st.markdown("""
<style>
    /* Global Background and Typography Overrides */
    .stApp {
        background-color: #0B0F19 !important;
    }
    html, body, [data-testid="stWidgetLabel"] p {
        color: #94A3B8 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 500;
        font-size: 14px;
    }
    h1 {
        color: #FFFFFF !important;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.05em !important;
    }
    h3 {
        color: #F1F5F9 !important;
        font-weight: 600 !important;
        letter-spacing: -0.03em !important;
    }
    
    /* Interactive Card Component Wrapper */
    div[data-testid="stForm"] {
        background-color: #111827 !important;
        border: 1px solid #1F2937 !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    }
    
    /* Premium Action Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        letter-spacing: -0.01em !important;
        width: 100% !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.3) !important;
    }
    
    /* Clean Divider Control */
    hr {
        border-color: #1F2937 !important;
    }
</style>
""", unsafe_html=True)

# ==============================================================================
# 2. SECURE COMPUTATIONAL ENGINE LOADING
# ==============================================================================
@st.cache_resource
def load_production_assets():
    with open('champion_xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_production_assets()
except FileNotFoundError:
    st.error("⚠️ Core Asset Sync Failure: Production files missing from execution path.")
    st.stop()

# ==============================================================================
# 3. INTERFACE FRAMEWORK & HEADER
# ==============================================================================
# Premium Header Layout
title_col, logo_col = st.columns([5, 1])
with title_col:
    st.title("AuraRisk // Credit Scoring OS")
    st.markdown("<p style='color: #64748B !important; font-size: 15px; margin-top: -10px;'>Risk assessment node powered by an Optimized Gradient Boosted Decision Tree pipeline.</p>", unsafe_html=True)

st.markdown("<br>", unsafe_html=True)

# Encapsulating inputs inside a clean unified Form layout
with st.form("risk_assessment_form"):
    st.subheader("📋 Core Underwriting Parameters")
    
    # Balanced 3-Column Interface Grid Arrangement
    col1, col2, col3 = st.columns(3)
    
    with col1:
        person_age = st.slider("Applicant Age (Years)", min_value=18, max_value=90, value=30)
        person_income = st.number_input("Annual Gross Income ($)", min_value=5000, max_value=500000, value=45000, step=1000)
        person_emp_length = st.slider("Employment Longevity (Years)", min_value=0, max_value=45, value=4)
        
    with col2:
        loan_amnt = st.number_input("Requested Principal Exposure ($)", min_value=1000, max_value=50000, value=10000, step=500)
        person_clean_int_rate = st.slider("Target Interest Rate Matrix (%)", min_value=4.0, max_value=25.0, value=11.5, step=0.1)
        home_ownership = st.selectbox("Residential Registry Class", ["MORTGAGE", "RENT", "OWN", "OTHER"])
        
    with col3:
        loan_intent = st.selectbox("Capital Allocation Purpose", ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE"])
        historical_default = st.selectbox("Historical Credit Bureau Default Flag", ["NO", "YES"])
        
    st.markdown("<br>", unsafe_html=True)
    submit_execution = st.form_submit_button("RUN AUTOMATED RISK ANALYSIS")

# ==============================================================================
# 4. PREPROCESSING & REAL-TIME INFERENCE PIPELINE
# ==============================================================================
if submit_execution:
    # 1. Real-time dynamic feature derivation
    loan_to_income_ratio = loan_amnt / person_income

    # 2. Structural Dictionary Initialization
    input_data = {
        'person_age': person_age,
        'person_income': person_income,
        'person_emp_length': person_emp_length,
        'loan_amnt': loan_amnt,
        'person_clean_int_rate': person_clean_int_rate,
        'loan_to_income_ratio': loan_to_income_ratio,
    }

    # 3. Handle One-Hot Encoding mapping precisely
    encoded_keys = [
        'person_home_ownership_OTHER', 'person_home_ownership_OWN', 'person_home_ownership_RENT',
        'loan_intent_EDUCATION', 'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE',
        'cb_person_default_on_file_Y'
    ]
    for key in encoded_keys:
        input_data[key] = 0

    # Activating structural tracking flags based on selection criteria
    if home_ownership != "MORTGAGE" and f"person_home_ownership_{home_ownership}" in input_data:
        input_data[f"person_home_ownership_{home_ownership}"] = 1

    if loan_intent != "EDUCATION" and f"loan_intent_{loan_intent}" in input_data:
        input_data[f"loan_intent_{loan_intent}"] = 1

    if historical_default == "YES":
        input_data['cb_person_default_on_file_Y'] = 1

    # Format into DataFrame array matching training sequence
    df_input = pd.DataFrame([input_data])
    expected_order = [
        'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
        'person_clean_int_rate', 'loan_to_income_ratio',
        'person_home_ownership_OTHER', 'person_home_ownership_OWN', 'person_home_ownership_RENT',
        'loan_intent_EDUCATION', 'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE',
        'cb_person_default_on_file_Y'
    ]
    df_input = df_input[expected_order]

    # Run robust feature normalization scaling safely without leakage
    numeric_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'person_clean_int_rate', 'loan_to_income_ratio']
    df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

    # 4. Generate Machine Learning Interpretations
    prediction = model.predict(df_input)[0]
    risk_probability = model.predict_proba(df_input)[0][1]

    # ==============================================================================
    # 5. METRIC PRESENTATION DISPLAY
    # ==============================================================================
    st.markdown("<hr>", unsafe_html=True)
    st.subheader("📊 Underwriting Decision Results")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if prediction == 0:
            st.markdown("""
            <div style='background-color: rgba(16, 185, 129, 0.1); border-left: 5px solid #10B981; padding: 1.5rem; border-radius: 8px;'>
                <h3 style='color: #10B981 !important; margin: 0; font-size: 20px;'>🟢 RISK METRICS: COMPLIANT</h3>
                <p style='color: #94A3B8; margin-top: 5px; margin-bottom: 0;'>Application passes institutional underwriting volatility thresholds.</p>
            </div>
            """, unsafe_html=True)
        else:
            st.markdown("""
            <div style='background-color: rgba(239, 68, 68, 0.1); border-left: 5px solid #EF4444; padding: 1.5rem; border-radius: 8px;'>
                <h3 style='color: #EF4444 !important; margin: 0; font-size: 20px;'>🔴 RISK METRICS: NON-COMPLIANT</h3>
                <p style='color: #94A3B8; margin-top: 5px; margin-bottom: 0;'>Application exhibits volatility risk signals tracking beyond acceptable levels.</p>
            </div>
            """, unsafe_html=True)
            
    with res_col2:
        st.markdown(f"""
        <div style='background-color: #111827; border: 1px solid #1F2937; padding: 1.15rem; border-radius: 8px;'>
            <span style='color: #64748B; font-size: 12px; text-transform: uppercase; letter-spacing: 0.1em;'>Modeled Default Probability</span>
            <h2 style='color: #FFFFFF; font-size: 32px; margin: 5px 0 10px 0; font-weight: 700;'>{risk_probability * 100:.2f}%</h2>
        </div>
        """, unsafe_html=True)
        st.progress(float(risk_probability))
