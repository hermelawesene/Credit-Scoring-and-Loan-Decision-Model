import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt # Added for static plotting

# --- Configuration for Caching ---
# Define a single function to load all objects and cache them
@st.cache_resource
def load_assets():
    """Loads the trained model, scaler, and encoders, and caches them."""
    try:
        # ------------------------
        # Load trained model & preprocessing objects
        # ------------------------
        with open("best_model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("scaler.pkl", "rb") as f: # Load your scaler
            scaler = pickle.load(f)

        with open("encoders.pkl", "rb") as f: # Load your label encoders
            encoders = pickle.load(f)

        return model, scaler, encoders
    except FileNotFoundError as e:
        st.error(f"Error: Missing asset file. Please ensure **{e.filename}** is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during asset loading: {e}")
        st.stop()

model, scaler, encoders = load_assets()

# --- Streamlit App ---
st.title("💳 Credit Risk Prediction App")
st.markdown("""
    Predict whether a loan applicant is likely to default (1) or not (0) based on their financial profile.
    
    *Note: This application requires the following files in the same directory: `best_model.pkl`, `scaler.pkl`, and `encoders.pkl`.*
""")

# ------------------------
# User Inputs (COLLECTING ALL 11 FEATURES)
# ------------------------
def user_input_features():
    st.sidebar.header("Applicant Details")
    person_age = st.sidebar.slider("Age", min_value=18, max_value=120, value=30)
    person_income = st.sidebar.number_input("Annual Income ($)", min_value=1000, value=50000, step=1000)
    person_emp_length = st.sidebar.number_input("Years Employed", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    
    st.sidebar.header("Loan Details")
    loan_amnt = st.sidebar.number_input("Loan Amount ($)", min_value=0, value=15000, step=1000)
    loan_int_rate = st.sidebar.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=30.0, value=12.0, step=0.1)
    
    st.sidebar.header("Credit History & Ownership")
    cb_person_cred_hist_length = st.sidebar.number_input("Credit History Length (years)", min_value=0, max_value=50, value=5)
    
    # Feature added based on the original dataset structure
    cb_person_default_on_file = st.sidebar.selectbox("Default on File?", ["Y", "N"])
    
    person_home_ownership = st.sidebar.selectbox("Home Ownership", ["MORTGAGE", "OTHER", "OWN", "RENT"])
    loan_intent = st.sidebar.selectbox("Loan Intent", ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "DEBTCONSOLIDATION", "VENTURE"])
    loan_grade = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])

    data = {
        "person_age": person_age,
        "person_income": person_income,
        "person_emp_length": person_emp_length,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "cb_person_default_on_file": cb_person_default_on_file, 
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display current inputs
st.subheader("Current Input Parameters")
st.dataframe(input_df, use_container_width=True)


# ------------------------
# Preprocess Input 
# ------------------------

# 1. Feature Engineering (Fix for loan_percent_income)
# Calculate the missing feature that caused the initial ValueError
# Handles potential division by zero by using a small epsilon if income is zero or very small
epsilon = 1e-6
input_df['loan_percent_income'] = input_df['loan_amnt'] / (input_df['person_income'] + epsilon)

# 2. Apply Label Encoders with Robustness Check
try:
    for col, le in encoders.items():
        if col in input_df.columns:
            val = input_df[col].iloc[0]
            # Check if the selected category is known to the encoder
            if val not in le.classes_:
                st.error(f"Configuration Error: The selected category '{val}' for {col} was not seen in the training data.")
                st.stop()
                
            input_df[col] = le.transform(input_df[col])
except ValueError as e:
    st.error(f"A data encoding error occurred. Please verify your asset files are correct. Details: {e}")
    st.stop()

# 3. One-hot encode multi-class features 
# The model requires the data to be in the final state used for training.
input_df = pd.get_dummies(input_df)

# Reindex to match training columns and fill missing OHE columns with 0
try:
    input_df = input_df.reindex(columns=model.get_booster().feature_names, fill_value=0)
except Exception as e:
    st.error(f"Reindexing Error: Failed to align input columns with model's expected features. Details: {e}")
    st.stop()


# 4. Scale numerical columns (UPDATED num_cols with the derived feature)
# CRITICAL: The order below must match the order in the training df_encoded header exactly.
num_cols = [
    'person_age',
    'person_income',
    'person_emp_length',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length' 
]

# Ensure all required columns are present and select them in the correct order
data_for_scaling = input_df[num_cols]

try:
    # Transform the data using the correct column order
    input_df[num_cols] = scaler.transform(data_for_scaling)
    st.success("Preprocessing complete. Ready for prediction.")
except ValueError as e:
    st.error("Scaling Order Error: The columns in the `num_cols` list are likely in the wrong order compared to your training script. Details: The feature names must be in the same order as they were in fit. Please verify the `num_cols` list.")
    st.stop()


# ------------------------
# Prediction & Explanation
# ------------------------
st.header("Results")
if st.button("Predict Credit Risk", type="primary"):
    with st.spinner("Calculating prediction..."):
        # Prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1] # Probability of Default (Class 1)

    st.subheader("Prediction Outcome")
    
    # Conditional formatting for the result
    if prediction == 1:
        st.error(f"Predicted Class: **Default**")
        st.metric(label="Probability of Default", value=f"{prediction_proba*100:.2f}%")
        st.markdown("⚠️ **High Risk:** The model suggests this applicant has a high likelihood of defaulting on the loan.")
    else:
        st.success(f"Predicted Class: **Non-Default**")
        st.metric(label="Probability of Default", value=f"{prediction_proba*100:.2f}%")
        st.balloons()
        st.markdown("✅ **Low Risk:** The model suggests this applicant is likely to repay the loan.")

    
    # SHAP Explainability
    st.subheader("Feature Importance (SHAP)")
    st.write("The chart below visualizes the impact of each feature on the final prediction.")
    
    try:
        explainer = shap.TreeExplainer(model)
        raw_shap_values = explainer.shap_values(input_df)
        
        # Determine the correct SHAP values and expected value based on the output structure
        if isinstance(raw_shap_values, list) and len(raw_shap_values) > 1:
            # Case: Output is [shap_class_0, shap_class_1]. We want class 1 (Default).
            shap_values = raw_shap_values[1]
            # Expected value may also be a list, so we handle it gracefully
            expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
        else:
            # Case: Output is a single array (already for the positive class).
            shap_values = raw_shap_values
            expected_value = explainer.expected_value
        
        # Generate the static waterfall plot using Matplotlib
        # This is more reliable in Streamlit than the interactive force_plot
        
        # 1. Create a SHAP Explanation object for the single prediction
        exp = shap.Explanation(
            values=shap_values[0],
            base_values=expected_value,
            data=input_df.iloc[0],
            feature_names=input_df.columns.tolist()
        )
        
        # 2. Use pyplot to generate the plot
        # Clear the current figure to ensure only the SHAP plot is displayed
        plt.clf() 
        shap.waterfall_plot(exp, show=False)
        
        # 3. Display the plot in Streamlit
        st.pyplot(plt.gcf())
        
    except Exception as e:
        st.warning(f"Could not generate SHAP plot. Ensure the SHAP library and model type are compatible: {e}")