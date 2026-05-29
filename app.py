import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==========================================
# 1. PREMIUM PAGE SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AuraRisk // Credit Scoring OS",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Minimalist CSS to give it a world-class, clean tech feel
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1 { color: #1e293b; font-family: 'Inter', sans-serif; font-weight: 800; }
    h3 { color: #334155; }
    .stButton>button {
        background-color: #2563eb; color: white; border-radius: 6px;
        padding: 0.5rem 2rem; font-weight: 600; width: 100%; border: none;
    }
    .stButton>button:hover { background-color: #1d4ed8; }
    </style>
""", unsafe_html=True)

# ==========================================
# 2. CACHED DATA & MODEL INGESTION
# ==========================================
@st.cache_resource
def load_assets():
    # Load our exported mathematical engines safely
    with open('champion_xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_assets()
except FileNotFoundError:
    st.error("⚠️ System Error: Asset files ('champion_xgb_model.pkl' or 'scaler.pkl') missing from server directory.")
    st.stop()

# ==========================================
# 3. INTERFACE HEADER
# ==========================================
st.title("💳 AuraRisk AI // Corporate Credit Scoring Engine")
st.caption("Enterprise-grade automated risk evaluation engine powered by Optimized Gradient Boosted Decision Trees.")
st.markdown("---")

# ==========================================
# 4. USER INPUT FIELDS (FORM MATRIX)
# ==========================================
st.subheader("📋 Applicant Information & Clinical Financial Profile")

# Create a clean 3-column structural layout for inputs
col1, col2, col3 = st.columns(3)

with col1:
    person_age = st.slider("Applicant Age", min_value=18, max_value=90, value=30)
    person_income = st.number_input("Annual Gross Income ($)", min_value=5000, max_value=500000, value=45000, step=1000)
    person_emp_length = st.slider("Employment Stability (Years)", min_value=0, max_value=45, value=4)

with col2:
    loan_amnt = st.number_input("Requested Loan Principal ($)", min_value=1000, max_value=50000, value=10000, step=500)
    person_clean_int_rate = st.slider("Offered Interest Rate (%)", min_value=4.0, max_value=25.0, value=11.5, step=0.1)
    
    home_ownership = st.selectbox("Residential Security", ["MORTGAGE", "RENT", "OWN", "OTHER"])

with col3:
    loan_intent = st.selectbox("Capital Allocation Allocation Purpose", ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE"])
    historical_default = st.selectbox("Historical Credit Bureau Default Flag", ["NO", "YES"])

# ==========================================
# 5. DATA PREPROCESSING & ALIGNMENT PIPELINE
# ==========================================
# 1. Dynamically engineer the core financial feature we taught our model
loan_to_income_ratio = loan_amnt / person_income

# 2. Assemble the base numeric values
input_data = {
    'person_age': person_age,
    'person_income': person_income,
    'person_emp_length': person_emp_length,
    'loan_amnt': loan_amnt,
    'person_clean_int_rate': person_clean_int_rate,
    'loan_to_income_ratio': loan_to_income_ratio,
}

# 3. Handle the one-hot encoded categories exactly how the model expects them
# Default all binary structural keys to 0
encoded_keys = [
    'person_home_ownership_OTHER', 'person_home_ownership_OWN', 'person_home_ownership_RENT',
    'loan_intent_EDUCATION', 'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE',
    'cb_person_default_on_file_Y'
]
for key in encoded_keys:
    input_data[key] = 0

# Set the active switches based on user selections
if home_ownership != "MORTGAGE" and f"person_home_ownership_{home_ownership}" in input_data:
    input_data[f"person_home_ownership_{home_ownership}"] = 1

if loan_intent != "EDUCATION" and f"loan_intent_{loan_intent}" in input_data:
    input_data[f"loan_intent_{loan_intent}"] = 1

if historical_default == "YES":
    input_data['cb_person_default_on_file_Y'] = 1

# Convert dictionary to DataFrame matching the exact training columns order
df_input = pd.DataFrame([input_data])

# Structural layout alignment fix
expected_order = [
    'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
    'person_clean_int_rate', 'loan_to_income_ratio',
    'person_home_ownership_OTHER', 'person_home_ownership_OWN', 'person_home_ownership_RENT',
    'loan_intent_EDUCATION', 'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE',
    'cb_person_default_on_file_Y'
]
df_input = df_input[expected_order]

# Scale numeric columns safely using the rules we stored inside scaler.pkl
numeric_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'person_clean_int_rate', 'loan_to_income_ratio']
df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

# ==========================================
# 6. INFERENCE & METRIC RISK METRICS DISPLAY
# ==========================================
st.markdown("---")
if st.button("RUN AUTOMATED RISK ANALYSIS"):
    # Generate predictive scores
    prediction = model.predict(df_input)[0]
    risk_probability = model.predict_proba(df_input)[0][1]
    
    st.subheader("📊 Underwriting Decision Results")
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if prediction == 0:
            st.success("🟢 APPLICATION APPROVED: LOW RISK PROFILE")
            st.metric(label="Calculated Credit Default Risk Assessment", value="PASS")
        else:
            st.error("🔴 APPLICATION DENIED: HIGH RISK OUTLIER")
            st.metric(label="Calculated Credit Default Risk Assessment", value="FAIL")
            
    with res_col2:
        st.metric(label="Model Probability Risk Score", value=f"{risk_probability * 100:.2f}%")
        st.progress(float(risk_probability))