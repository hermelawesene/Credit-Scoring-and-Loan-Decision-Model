Credit Scoring and Loan Decision Model
-----------------------------------------

This project implements a machine learning pipeline to predict credit risk (loan default status) using an XGBoost Classifier. It includes a Streamlit web application for real-time predictions and integrates SHAP (SHapley Additive exPlanations) for transparent, explainable decision-making.

###  Key Features

*   **Credit Risk Prediction:** Uses a trained XGBoost model to predict the likelihood of a borrower defaulting (1) or not defaulting (0).
    
*   **Real-time Web App:** A user-friendly Streamlit interface (app\_corrected\_shap.py) allows interactive input of applicant and loan details.
    
*   **Explainability (SHAP):** Provides a static SHAP Waterfall Plot to explain _why_ the model made a specific prediction for an individual applicant, highlighting which features pushed the prediction toward "Default" or "Non-Default."
    
*   **Robust Fairness Analysis:** Includes a dedicated script (fairness\_analysis.py) to test for bias against protected or proxy attributes, ensuring compliance with fair lending principles.
    

###  Technology Stack

*   **Core Model:** Python, Scikit-learn, XGBoost
    
*   **Data Handling:** Pandas, NumPy
    
*   **Explainability:** SHAP
    
*   **Web Application:** Streamlit
    
*   **Visualization:** Matplotlib
    

###  Setup and Installation

Follow these steps to set up the project locally and run the Streamlit application.

#### 1\. Prerequisites

You must have Python 3.8+ installed.

#### 2\. Clone the Repository
   `git clone https://github.com/hermelawesene/Credit-Scoring-and-Loan-Decision-Model.git  
   cd Credit-Scoring-and-Loan-Decision-Model   `

#### 3\. Create and Activate Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

   `# Create the environment  python -m venv .venv   # Activate the environment (Windows)  .\.venv\Scripts\activate  # Activate the environment (macOS/Linux)  source .venv/bin/activate   `

#### 4\. Install Dependencies

`   pip install -r requirements.txt   # Note: You may need to create a requirements.txt file with:  # streamlit  # pandas  # numpy  # scikit-learn  # xgboost  # shap  # matplotlib   `

### 📦 Data and Model Assets (CRITICAL)

The Streamlit application relies on three pre-trained files generated during the model training phase. **These files MUST be present in the project's root directory to run the application:**

1.  best\_model.pkl: The saved XGBoostClassifier object.
    
2.  scaler.pkl: The fitted MinMaxScaler object used for numerical feature scaling.
    
3.  encoders.pkl: A dictionary containing fitted LabelEncoder objects for binary categorical features.
    

### ▶️ Running the Application

To start the interactive prediction app, run the following command from the root directory:

`   streamlit run app_corrected_shap.py   `

The application will open in your web browser.

### ⚖️ Fairness and Bias Analysis Summary

A critical analysis was performed using the fairness\_analysis.py script to audit the model for bias across key groups.

Attribute Tested

Comparison

Metric

Result

Finding

**Age**

Young (18-25) vs. Prime (26-55)

Disparate Impact Ratio (DIR)

**0.988**

**PASS:** Approval rates are nearly identical.

**Age**

Young (18-25) vs. Prime (26-55)

Equal Opportunity Difference (EOD)

**0.002**

**PASS:** Model performance on truly creditworthy applicants is equal.

**Home Ownership**

Rent (Protected Proxy) vs. Mortgage (Reference)

Disparate Impact Ratio (DIR)

**0.669**

**FAIL:** Significant disparate impact detected.

**Mitigation Required:** The model demonstrates bias against applicants who currently **Rent** their homes (DIR is below the 0.8 regulatory threshold). Future work must focus on mitigating this bias through techniques like model re-weighting or threshold adjustment to ensure fair lending practices.
