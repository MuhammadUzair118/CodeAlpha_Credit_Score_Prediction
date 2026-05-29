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

# Premium HTML style injections for clean, modern enterprise typography and layout
st.html("""
<style>
    /* Global Base Canvas */
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
    
    /* Premium Corporate Action Button (Stripe Style Blue) */
    .stButton>button {
        background: #0284C7 !important;
        color: #FFFFFF !important;
        border: 1px solid #0369A1 !important;
        border-radius: 6px !important;
        padding: 0.85rem 2rem !important;
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
    
    /* Premium Style Overrides for Streamlit Metrics Component */
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #0F172A !important;
    }
    [data-testid="stMetricLabel"] p {
        font-size: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #64748B !important;
    }
    
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
    loan_to_income_ratio = float(loan_amnt / person_income)

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

    if home_ownership != "MORTGAGE" and f"person_home_ownership_{home_ownership}" in input_data:
        input_data[f"person_home_ownership_{home_ownership}"] = 1

    if loan_intent != "EDUCATION" and f"loan_intent_{loan_intent}" in input_data:
        input_data[f"loan_intent_{loan_intent}"] = 1

    if historical_default == "YES":
        input_data['cb_person_default_on_file_Y'] = 1

    df_inference = pd.DataFrame([input_data])

    ordered_columns = [
        'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
        'person_clean_int_rate', 'loan_to_income_ratio',
        'person_home_ownership_OTHER', 'person_home_ownership_OWN', 'person_home_ownership_RENT',
        'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE',
        'cb_person_default_on_file_Y'
    ]
    df_inference = df_inference[ordered_columns]

    numeric_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'person_clean_int_rate', 'loan_to_income_ratio']
    df_inference[numeric_features] = scaler.transform(df_inference[numeric_features])

    final_matrix_values = df_inference.values
    prediction = model.predict(final_matrix_values)[0]
    risk_probability = model.predict_proba(final_matrix_values)[0][1]

    # ==============================================================================
    # 5. VISUAL REPORTING ENGAGEMENT PANEL (LINKEDIN READY)
    # ==============================================================================
    st.html("<hr>")
    st.subheader("📊 Institutional Risk Analysis Overview")
    
    # Financial Analytics Cards Block
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(label="Requested Principal", value=f"${loan_amnt:,.0f}")
    with m_col2:
        st.metric(label="Debt-to-Income Index", value=f"{loan_to_income_ratio * 100:.1f}%")
    with m_col3:
        st.metric(label="Assigned Matrix Rate", value=f"{person_clean_int_rate:.2f}%")
    with m_col4:
        st.metric(label="Calculated Default Risk", value=f"{risk_probability * 100:.2f}%")

    st.html("<br>")

    # Split Output Section
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        if prediction == 0:
            st.html("""
            <div style='background-color: #F0FDF4; border: 1px solid #BBF7D0; border-left: 5px solid #16A34A; padding: 1.75rem; border-radius: 8px;'>
                <h3 style='color: #16A34A !important; margin: 0; font-size: 20px; font-weight:700;'>🟢 UNDERWRITING DECISION: COMPLIANT</h3>
                <p style='color: #475569; margin-top: 8px; margin-bottom: 0; font-weight: 400; line-height:1.5;'>
                    The applicant's financial velocity profile maps cleanly within secure volatility bounds. This application is cleared for automated corporate capital allocation.
                </p>
            </div>
            """)
        else:
            st.html("""
            <div style='background-color: #FEF2F2; border: 1px solid #FEE2E2; border-left: 5px solid #DC2626; padding: 1.75rem; border-radius: 8px;'>
                <h3 style='color: #DC2626 !important; margin: 0; font-size: 20px; font-weight:700;'>🔴 UNDERWRITING DECISION: REJECTED</h3>
                <p style='color: #475569; margin-top: 8px; margin-bottom: 0; font-weight: 400; line-height:1.5;'>
                    The predictive engine flagged significant risk anomalies. Credit metrics exceed acceptable probability limits for automated institutional exposure.
                </p>
            </div>
            """)
            
    with res_col2:
        # High-End Visual Distribution Chart Component using Native Streamlit Horizontal Bar Framework
        st.html("<div style='margin-bottom: 4px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; font-weight:600;'>Risk Spectrum Allocation Chart</div>")
        
        # Constructing a premium visualization dataset to display threshold bars on screen
        chart_data = pd.DataFrame({
            'Risk Vector': ['Calculated Probability', 'Institutional Alert Threshold'],
            'Percentage (%)': [float(risk_probability * 100), 25.0] # 25.0% acts as a clear standard enterprise baseline
        })
        
        st.bar_chart(
            data=chart_data,
            x='Risk Vector',
            y='Percentage (%)',
            color=['#0284C7' if prediction == 0 else '#DC2626'], # Adapts color instantly to fit the risk profile status
            use_container_width=True,
            height=160
        )
