import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# --- Configuration ---
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, 'multi_output_model.joblib')
ENCODERS_PATH = os.path.join(MODEL_DIR, 'label_encoders.joblib')
SELECTOR_PATH = os.path.join(MODEL_DIR, 'feature_selector.joblib')
INVERSE_MAPS_PATH = os.path.join(MODEL_DIR, 'inverse_maps.joblib')
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, 'feature_cols.joblib')

# --- Load Artifacts ---
@st.cache_resource
def load_model_artifacts():
    try:
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"Model directory '{MODEL_DIR}' not found.")
        
        model = joblib.load(MODEL_PATH)
        label_encoders = joblib.load(ENCODERS_PATH)
        selector = joblib.load(SELECTOR_PATH)
        inverse_maps = joblib.load(INVERSE_MAPS_PATH)
        original_feature_cols = joblib.load(FEATURE_COLS_PATH)
        
        return model, label_encoders, selector, inverse_maps, original_feature_cols
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()

# --- Initialize Streamlit App ---
st.set_page_config(
    page_title="Demographic Predictor",
    page_icon="📊",
    layout="wide"
)

# --- App Title and Description ---
st.title("Demographic Predictor")
st.markdown("""
    Enter spending habit indicators to predict user demographics.
    Select values for each category to get predictions for Gender, Age Group, Occupation, and Income Level.
""")

# --- Load Model Artifacts ---
model, label_encoders, selector, inverse_maps, original_feature_cols = load_model_artifacts()

# --- Define Form Fields and Options ---
form_fields = [
    {"name": "Period", "label": "Period"},
    {"name": "Location", "label": "Location"},
    {"name": "Personal_Hygiene__eg_Soap__toothpaste_", "label": "Personal Hygiene", "alias": "Personal Hygiene (eg Soap, toothpaste)"},
    {"name": "Cleaning_products", "label": "Cleaning Products", "alias": "Cleaning products"},
    {"name": "Long_lasting__dry__groceries", "label": "Long Lasting Groceries", "alias": "Long lasting (dry) groceries"},
    {"name": "Fresh_groceries__Fruits__vegetables_", "label": "Fresh Groceries", "alias": "Fresh groceries (Fruits, vegetables)"},
    {"name": "Medicines_Natural_remedies", "label": "Medicines/Remedies", "alias": "Medicines/Natural remedies"},
    {"name": "Alcohol_beverages", "label": "Alcohol Beverages", "alias": "Alcohol beverages"},
    {"name": "Skin_care__eg__Body_lotion_", "label": "Skin Care", "alias": "Skin care (eg. Body lotion)"},
    {"name": "Hair_care__eg__Shampoo_", "label": "Hair Care", "alias": "Hair care (eg. Shampoo)"},
    {"name": "Entertainment__eg__Restaurants__movies_", "label": "Entertainment", "alias": "Entertainment (eg. Restaurants, movies)"},
    {"name": "Electronics__eg_Phone__Computers_", "label": "Electronics", "alias": "Electronics (eg Phone, Computers)"},
    {"name": "Beauty__eg_Makeup__cosmetics__haircuts_", "label": "Beauty", "alias": "Beauty (eg Makeup, cosmetics, haircuts)"},
    {"name": "Clothing", "label": "Clothing"},
    {"name": "Airtime_Data_bundles", "label": "Airtime/Data", "alias": "Airtime/Data bundles"}
]

dropdown_options = {
    field["name"]: ["More", "Less", "Same", "Not sure"] for field in form_fields
}
dropdown_options["Period"] = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"]
dropdown_options["Location"] = ["MS1", "MS2", "MS3", "MS4", "MS5"]

# --- Create Input Form ---
with st.form("prediction_form"):
    st.subheader("Input Features")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    # Place form fields in columns
    for i, field in enumerate(form_fields):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.selectbox(
                label=field["label"],
                options=dropdown_options[field["name"]],
                key=field["name"],
                index=0
            )
    
    submit_button = st.form_submit_button("Predict")

# --- Process Prediction ---
if submit_button:
    try:
        # 1. Collect input data
        input_data = {}
        for field in form_fields:
            field_name = field["name"]
            alias = field.get("alias", field_name)
            input_data[alias] = st.session_state[field_name]
        
        # 2. Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # 3. Ensure correct column order
        input_df = input_df[original_feature_cols]
        
        # 4. Apply Label Encoding
        input_encoded = input_df.copy()
        for col, le in label_encoders.items():
            if col in input_encoded.columns:
                current_col_data = input_encoded[col].astype(str)
                known_classes = set(le.classes_)
                unknown_marker = -1
                encoded_values = []
                
                for item in current_col_data:
                    if item in known_classes:
                        encoded_values.append(le.transform([item])[0])
                    else:
                        encoded_values.append(unknown_marker)
                
                input_encoded[col] = encoded_values
        
        # 5. Apply Feature Selection
        input_selected = selector.transform(input_encoded)
        
        # 6. Make Prediction
        prediction_encoded = model.predict(input_selected)
        prediction_values = prediction_encoded[0]
        
        # 7. Inverse Transform Predictions
        expected_target_order = ['Gender', 'Age-group', 'Occupation', 'Income Level']
        results = {}
        
        for i, target_name in enumerate(expected_target_order):
            encoded_val = prediction_values[i]
            inv_map = inverse_maps.get(target_name)
            decoded_val = inv_map.get(encoded_val, f"Unknown code ({encoded_val})")
            results[target_name] = decoded_val
        
        # 8. Display Results
        st.subheader("Prediction Results")
        st.markdown("""
            | Category | Prediction |
            |---------|------------|
        """)
        for target, value in results.items():
            st.markdown(f"| {target} | {value} |")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}") 