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

# Native HTML Injection for a clean, premium enterprise white/slate layout
st.html("""
<style>
    /* Global Base Canvas - Pure White and Professional Slate Gray */
    .stApp {
        background-color: #FFFFFF !important;
    }
    html, body, [data-testid="stWidgetLabel"] p {
        color: #334155 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        font-weight: 550 !important;
        font-size: 14px;
    }
    
    /* Typography Hierarchy */
    h1 {
        color: #0F172A !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.04em !important;
    }
    h3 {
        color: #1E293B !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Clean Enterprise Form Container */
    div[data-testid="stForm"] {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        padding: 2.5rem !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Premium Corporate Action Button (Stripe/Mercury Style Blue) */
    .stButton>button {
        background: #0284C7 !important;
        color: #FFFFFF !important;
        border: 1px solid #0369A1 !important;
        border-radius: 6px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        letter-spacing: -0.01em !important;
        width: 100% !important;
        transition: all 0.15s ease-in-out !important;
    }
    .stButton>button:hover {
        background: #0369A1 !important;
        border-color: #075985 !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.15) !important;
    }
    
    /* Structural Dividers */
    hr {
        border-color: #E2E8F0 !important;
    }
</style>
""")

# ==============================================================================
# 2. ASSET SYNCHRONIZATION PIPELINE
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
    st.error("System Error: Critical mathematical model assets missing from the server root directory.")
    st.stop()

# ==============================================================================
# 3. INTERFACE FRAMEWORK & APPLICATION DATA FIELDS
# ==============================================================================
title_col, logo_col = st.columns([5, 1])
with title_col:
    st.title("AuraRisk Terminal // Credit Risk Analytics Engine")
    st.html("<p style='color: #64748B !important; font-size: 15px; margin-top: -10px; font-weight: 400;'>Institutional-grade machine learning model for automated capital allocation risk profiling.</p>")

st.html("<br>")

with st.form("underwriting_assessment_form"):
    st.subheader("📋 Empirical Risk Metrics Input Matrix")
    
    # 3-Column Structural Grid
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
# 4. DETERMINISTIC FEATURE ENGINEERING & RIGID ALIGNMENT MATRIX
# ==============================================================================
if submit_execution:
    # 1. Engineered Metric Derivation
    loan_to_income_ratio = float(loan_amnt / person_income)

    # 2. Build structured dictionary with the exact 13 verified columns from diagnostics
    input_data = {
        'person_age': float(person_age),
        'person_income': float(person_income),
        'person_emp_length': float(person_emp_length),
        'loan_amnt': float(loan_amnt),
        'person_clean_int_rate': float(person_clean_int_rate),
        'loan_to_income_ratio': loan_to_income_ratio,
        'person_home_ownership_OTHER': 0,
        'person_home_ownership_OWN': 0,
        'person_home_ownership_RENT': 0,
        'loan_intent_MEDICAL': 0,
        'loan_intent_PERSONAL': 0,
        'loan_intent_VENTURE': 0,
        'cb_person_default_on_file_Y': 0
    }

    # 3. Trigger structural hot-encoded indicators based on runtime selection
    if home_ownership != "MORTGAGE" and f"person_home_ownership_{home_ownership}" in input_data:
        input_data[f"person_home_ownership_{home_ownership}"] = 1

    # (Note: If user selects 'EDUCATION', all intent switches naturally remain 0, matching the baseline)
    if loan_intent != "EDUCATION" and f"loan_intent_{loan_intent}" in input_data:
        input_data[f"loan_intent_{loan_intent}"] = 1

    if historical_default == "YES":
        input_data['cb_person_default_on_file_Y'] = 1

    # Convert to DataFrame
    df_inference = pd.DataFrame([input_data])

    # Enforce precise verified order expected by the binary booster trees
    ordered_columns = [
        'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
        'person_clean_int_rate', 'loan_to_income_ratio',
        'person_home_ownership_OTHER', 'person_home_ownership_OWN', 'person_home_ownership_RENT',
        'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE',
        'cb_person_default_on_file_Y'
    ]
    df_inference = df_inference[ordered_columns]

    # 4. Scale continuous features subset securely
    numeric_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'person_clean_int_rate', 'loan_to_income_ratio']
    df_inference[numeric_features] = scaler.transform(df_inference[numeric_features])

    # 5. Extract matrix values array (Shape: 1, 13) to feed the engine
    final_matrix_values = df_inference.values
    
    prediction = model.predict(final_matrix_values)[0]
    risk_probability = model.predict_proba(final_matrix_values)[0][1]

    # ==============================================================================
    # 5. ENTERPRISE REPORTING METRICS PANEL
    # ==============================================================================
    st.html("<hr>")
    st.subheader("📊 Institutional Underwriting Decision Results")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if prediction == 0:
            st.html("""
            <div style='background-color: #F0FDF4; border-left: 4px solid #16A34A; padding: 1.5rem; border-radius: 6px;'>
                <h3 style='color: #16A34A !important; margin: 0; font-size: 18px;'>🟢 APPLICATION STATUS: APPROVED</h3>
                <p style='color: #475569; margin-top: 5px; margin-bottom: 0; font-weight: 400;'>Risk metrics settle within institutional default safety thresholds.</p>
            </div>
            """)
        else:
            st.html("""
            <div style='background-color: #FEF2F2; border-left: 4px solid #DC2626; padding: 1.5rem; border-radius: 6px;'>
                <h3 style='color: #DC2626 !important; margin: 0; font-size: 18px;'>🔴 APPLICATION STATUS: REJECTED</h3>
                <p style='color: #475569; margin-top: 5px; margin-bottom: 0; font-weight: 400;'>Risk metrics expose extreme volatility signatures violating safety baselines.</p>
            </div>
            """)
            
    with res_col2:
        st.html(f"""
        <div style='background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 1.15rem; border-radius: 6px;'>
            <span style='color: #64748B; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;'>Modeled Default Probability</span>
            <h2 style='color: #0F172A; font-size: 30px; margin: 2px 0 8px 0; font-weight: 700;'>{risk_probability * 100:.2f}%</h2>
        </div>
        """)
        st.progress(float(risk_probability))
