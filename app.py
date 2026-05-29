import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==============================================================================
# 1. INSTITUTIONAL TECH MINIMALIST UI ARCHITECTURE (WHITE & BLUE)
# ==============================================================================
st.set_page_config(
    page_title="AuraRisk Terminal // Credit Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.html("""
<style>
    /* Global Base Canvas */
    .stApp { background-color: #FFFFFF !important; }
    html, body, [data-testid="stWidgetLabel"] p {
        color: #334155 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        font-weight: 550 !important;
        font-size: 14px;
    }
    h1 { color: #0F172A !important; font-weight: 700 !important; letter-spacing: -0.04em !important; }
    h3 { color: #1E293B !important; font-weight: 600 !important; }
    
    /* Clean Enterprise Container */
    div[data-testid="stForm"] {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        padding: 2.5rem !important;
    }
    
    /* Premium Action Button */
    .stButton>button {
        background: #0284C7 !important;
        color: #FFFFFF !important;
        border: 1px solid #0369A1 !important;
        border-radius: 6px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
    }
</style>
""")

# ==============================================================================
# 2. ASSET LOADING & AUTOMATED DIAGNOSTICS
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
    st.error("System Error: Critical mathematical model assets missing.")
    st.stop()

# ==============================================================================
# 3. INTERFACE FRAMEWORK & APPLICATION DATA FIELDS
# ==============================================================================
st.title("AuraRisk Terminal // Credit Risk Analytics Engine")
st.html("<p style='color: #64748B !important; font-size: 15px; margin-top: -10px; font-weight: 400;'>Institutional-grade machine learning model diagnostics.</p>")

# --- DIAGNOSTICS PANEL (CRITICAL BROADCAST) ---
st.markdown("### 🔍 Model Architecture Diagnostics")
col_diag1, col_diag2 = st.columns(2)

with col_diag1:
    try:
        # Extracting the absolute feature names the XGBoost booster was trained on
        booster_features = model.get_booster().feature_names
        if booster_features:
            st.info(f"📋 **Model Expected Features ({len(booster_features)} total):**")
            st.code(str(booster_features))
        else:
            # If features were passed as an unnamed array during training
            st.warning(f"⚠️ **Model expects exactly {model.n_features_in_} features**, but they have no text names (trained as raw matrix).")
    except Exception as e:
        st.error(f"Could not read model feature names: {e}")

with col_diag2:
    try:
        # Inspecting the feature footprint of your scaler file
        scaler_features = scaler.n_features_in_
        st.info(f"📐 **Scaler Expected Input Features:** {scaler_features}")
    except Exception as e:
        st.error(f"Could not read scaler attributes: {e}")

st.markdown("---")

with st.form("underwriting_assessment_form"):
    st.subheader("📋 Empirical Risk Metrics Input Matrix")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        person_age = st.slider("Applicant Age (Years)", min_value=18, max_value=90, value=30)
        person_income = st.number_input("Annual Gross Income (USD)", min_value=5000, max_value=500000, value=45000, step=1000)
        person_emp_length = st.slider("Employment Stability (Years)", min_value=0, max_value=45, value=4)
        
    with col2:
        loan_amnt = st.number_input("Requested Principal Capital Exposure (USD)", min_value=1000, max_value=50000, value=10000, step=500)
        person_clean_int_rate = st.slider("Offered Interest Rate Matrix (%)", min_value=4.0, max_value=25.0, value=11.5, step=0.1)
        home_ownership = st.selectbox("Residential Security Class", ["MORTGAGE", "RENT", "OWN", "OTHER"])
        
    with col3:
        loan_intent = st.selectbox("Capital Allocation Purpose", ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE"])
        historical_default = st.selectbox("Historical Credit Bureau Default Status", ["NO", "YES"])
        
    st.html("<br>")
    submit_execution = st.form_submit_button("EVALUATE DEBT RISK PROFILE")

# ==============================================================================
# 4. PREPROCESSING PIPELINE
# ==============================================================================
if submit_execution:
    loan_to_income_ratio = float(loan_amnt / person_income)

    input_data = {
        'person_age': float(person_age), 'person_income': float(person_income),
        'person_emp_length': float(person_emp_length), 'loan_amnt': float(loan_amnt),
        'person_clean_int_rate': float(person_clean_int_rate), 'loan_to_income_ratio': loan_to_income_ratio,
        'person_home_ownership_OTHER': 0, 'person_home_ownership_OWN': 0, 'person_home_ownership_RENT': 0,
        'loan_intent_EDUCATION': 0, 'loan_intent_MEDICAL': 0, 'loan_intent_PERSONAL': 0, 'loan_intent_VENTURE': 0,
        'cb_person_default_on_file_Y': 0
    }

    if home_ownership != "MORTGAGE" and f"person_home_ownership_{home_ownership}" in input_data:
        input_data[f"person_home_ownership_{home_ownership}"] = 1
    if loan_intent != "EDUCATION" and f"loan_intent_{loan_intent}" in input_data:
        input_data[f"loan_intent_{loan_intent}"] = 1
    if historical_default == "YES":
        input_data['cb_person_default_on_file_Y'] = 1

    df_inference = pd.DataFrame([input_data])
    
    # Structural execution printout to screen during error state tracking
    st.write("Current App Hardcoded Features Structure Shape:", df_inference.shape)

    try:
        numeric_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'person_clean_int_rate', 'loan_to_income_ratio']
        df_inference[numeric_features] = scaler.transform(df_inference[numeric_features])
        
        final_matrix_values = df_inference.values
        prediction = model.predict(final_matrix_values)[0]
        st.success(f"Execution Output: {prediction}")
    except Exception as error_msg:
        st.error(f"Prediction Pipeline Crash: {error_msg}")
