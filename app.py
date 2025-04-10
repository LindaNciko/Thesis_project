import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import tempfile
import urllib.request
import gdown
from pathlib import Path
import time

# --- Initialize Streamlit App ---
# This MUST be the first Streamlit command
st.set_page_config(
    page_title="Demographic Predictor",
    page_icon="📊",
    layout="wide"
)

# --- Configuration ---
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, 'multi_output_model.joblib')
ENCODERS_PATH = os.path.join(MODEL_DIR, 'label_encoders.joblib')
SELECTOR_PATH = os.path.join(MODEL_DIR, 'feature_selector.joblib')
INVERSE_MAPS_PATH = os.path.join(MODEL_DIR, 'inverse_maps.joblib')
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, 'feature_cols.joblib')

# --- Environment Variables for Cloud Deployment ---
# Check for deployment environment - either from environment variable or auto-detect
DEPLOYMENT_ENV = os.environ.get("STREAMLIT_DEPLOYMENT", "")
if not DEPLOYMENT_ENV and os.environ.get("STREAMLIT_RUNTIME_EXISTS"):
    # Auto-detect if we're running on Streamlit Cloud
    DEPLOYMENT_ENV = "cloud"

# Get file ID from secrets - handle missing secrets gracefully
try:
    SELECTOR_GDRIVE_URL = st.secrets.get("SELECTOR_GDRIVE_URL", "")
except (FileNotFoundError, Exception) as e:
    st.warning("⚠️ No secrets file found. Running in local development mode.")
    SELECTOR_GDRIVE_URL = ""

# Show development mode notice
if not DEPLOYMENT_ENV and not SELECTOR_GDRIVE_URL:
    st.info("🖥️ Running in local development mode - will attempt to load model files directly from disk.")

# --- Helper Functions ---
def download_from_gdrive(file_id, output_path, show_progress=True):
    """Download a file from Google Drive with progress indicator"""
    if show_progress:
        with st.status("Downloading large model file...") as status:
            st.write("This may take a few minutes...")
            progress_bar = st.progress(0)
            
            # Create a function to update progress
            def callback(value):
                if isinstance(value, int) or isinstance(value, float):
                    progress_bar.progress(min(value/100, 1.0))
                
            start_time = time.time()
            success = gdown.download(
                f"https://drive.google.com/uc?id={file_id}", 
                output_path, 
                quiet=False,
                fuzzy=True,
                speed=callback
            )
            elapsed = time.time() - start_time
            
            if success:
                status.update(label=f"✅ Download complete in {elapsed:.1f} seconds", state="complete")
                return True
            else:
                status.update(label="❌ Download failed", state="error")
                return False
    else:
        return gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=True)

# --- Load Artifacts ---
@st.cache_resource(show_spinner="Loading model artifacts...")
def load_model_artifacts():
    """Load all model artifacts with comprehensive error handling"""
    try:
        # Create model directory if it doesn't exist
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR, exist_ok=True)
            st.info(f"Created model directory: {MODEL_DIR}")
        
        # Dictionary to track loading status
        artifacts = {
            "model": {"path": MODEL_PATH, "required": True, "loaded": False},
            "label_encoders": {"path": ENCODERS_PATH, "required": True, "loaded": False},
            "inverse_maps": {"path": INVERSE_MAPS_PATH, "required": True, "loaded": False},
            "feature_cols": {"path": FEATURE_COLS_PATH, "required": True, "loaded": False},
            "selector": {"path": SELECTOR_PATH, "required": True, "loaded": False}
        }
        
        # Load smaller artifacts first
        results = {}
        for name, info in artifacts.items():
            if name != "selector":  # Handle selector separately
                if os.path.exists(info["path"]):
                    try:
                        results[name] = joblib.load(info["path"])
                        artifacts[name]["loaded"] = True
                    except Exception as e:
                        st.warning(f"Error loading {name}: {str(e)}")
                        if info["required"]:
                            raise Exception(f"Failed to load required artifact: {name}")
                else:
                    st.warning(f"Missing artifact: {name} at path {info['path']}")
                    if info["required"]:
                        raise Exception(f"Required artifact not found: {name}")
        
        # Handle the large selector file differently depending on environment
        if DEPLOYMENT_ENV == "cloud":
            st.info("🌐 Running in cloud environment")
            
            # Check if the file ID is provided
            if not SELECTOR_GDRIVE_URL:
                st.error("⚠️ Missing Google Drive file ID for feature selector")
                st.info("Please set SELECTOR_GDRIVE_URL in Streamlit secrets")
                raise Exception("Feature selector URL not configured")
            
            # Create a temporary file for downloading
            with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as temp_file:
                temp_path = temp_file.name
            
            # Download the file from Google Drive
            if download_from_gdrive(SELECTOR_GDRIVE_URL, temp_path):
                try:
                    # Load the downloaded model
                    selector = joblib.load(temp_path)
                    results["selector"] = selector
                    artifacts["selector"]["loaded"] = True
                    # Clean up the temp file
                    os.unlink(temp_path)
                except Exception as e:
                    st.error(f"Failed to load downloaded selector: {str(e)}")
                    raise
            else:
                st.error("Failed to download feature selector from Google Drive")
                st.info("Check if the file ID is correct and the file is accessible")
                raise Exception("Download failed")
        else:
            # Local development - load directly
            st.info("🖥️ Running in local environment")
            selector_path = artifacts["selector"]["path"]
            if os.path.exists(selector_path):
                try:
                    results["selector"] = joblib.load(selector_path)
                    artifacts["selector"]["loaded"] = True
                except Exception as e:
                    st.error(f"Error loading selector: {str(e)}")
                    raise
            else:
                # Special handling for missing feature selector in development mode
                if not os.path.exists(MODEL_DIR):
                    st.error(f"Model directory not found: {MODEL_DIR}")
                    st.info("Please create the model directory and add required model files")
                    raise FileNotFoundError(f"Model directory not found")
                
                st.warning(f"⚠️ Feature selector file not found at: {selector_path}")
                st.info("""
                For local development without the large model file, you have two options:
                
                1. Get the model file:
                   - Download the feature_selector.joblib file
                   - Place it in the model/ directory
                   
                2. Upload to Google Drive:
                   - Upload the file to Google Drive
                   - Share it with "Anyone with the link"
                   - Get the file ID and add it to .streamlit/secrets.toml
                """)
                
                # Create a simple fallback selector if possible
                try:
                    # The selected features file should be available and is small
                    selected_features_path = os.path.join(MODEL_DIR, 'selected_features.joblib')
                    if os.path.exists(selected_features_path):
                        # Try to create a simple selector based on the selected features
                        from sklearn.feature_selection import SelectKBest
                        
                        selected_features = joblib.load(selected_features_path)
                        
                        class SimpleSelector:
                            def __init__(self, feature_names):
                                self.feature_names = feature_names
                                
                            def transform(self, X):
                                # Return only the features that were selected
                                return X[self.feature_names] if hasattr(X, 'iloc') else X
                        
                        # Create a simple selector using the selected feature names
                        results["selector"] = SimpleSelector(selected_features)
                        artifacts["selector"]["loaded"] = True
                        
                        st.success("✅ Created simplified feature selector for development")
                    else:
                        raise FileNotFoundError(f"Selected features file not found: {selected_features_path}")
                except Exception as e:
                    st.error(f"Could not create fallback selector: {str(e)}")
                    raise Exception(f"Selector file not found and could not create fallback: {str(e)}")
        
        # Validate that all required artifacts are loaded
        missing = [name for name, info in artifacts.items() if info["required"] and not info["loaded"]]
        if missing:
            raise Exception(f"Failed to load required artifacts: {', '.join(missing)}")
        
        # Return the loaded artifacts in the expected order
        return (
            results["model"],
            results["label_encoders"],
            results["selector"],
            results["inverse_maps"],
            results["feature_cols"]
        )
    except Exception as e:
        st.error(f"Error loading model artifacts: {str(e)}")
        
        # Show troubleshooting information
        with st.expander("Troubleshooting Information"):
            st.markdown("""
            ### Troubleshooting Steps
            
            1. **For local development**:
               - Ensure all model files are in the `model/` directory
               - Check file permissions
               
            2. **For Streamlit Cloud deployment**:
               - Verify SELECTOR_GDRIVE_URL is set in Streamlit secrets
               - Confirm the Google Drive file is publicly accessible
               - Check that STREAMLIT_DEPLOYMENT environment variable is set to "cloud"
            
            3. **Common issues**:
               - File corruption during upload
               - Incorrect file ID in secrets
               - Google Drive rate limiting
            """)
        
        st.stop()

# --- App Title and Description ---
st.title("Demographic Predictor")
st.markdown("""
    Enter spending habit indicators to predict user demographics.
    Select values for each category to get predictions for Gender, Age Group, Occupation, and Income Level.
""")

# --- Load Model Artifacts ---
with st.spinner("Loading model artifacts..."):
    model, label_encoders, selector, inverse_maps, original_feature_cols = load_model_artifacts()
    st.success("✅ Models loaded successfully")

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