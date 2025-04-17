# ========== app.py (Modified) ==========
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
# import tempfile # Not needed if not downloading selector
# import urllib.request
# import gdown # Not needed if not downloading selector
from pathlib import Path
import time
import traceback

# --- Initialize Streamlit App ---
st.set_page_config(
    page_title="Demographic Predictor (All Features)", # Updated title
    page_icon="📊",
    layout="wide"
)

# --- Configuration ---
MODEL_DIR = Path("model")
# === LOAD THE NEW MODEL FILE ===
MODEL_PATH = MODEL_DIR / 'multi_output_model_all_features.joblib'
ENCODERS_PATH = MODEL_DIR / 'label_encoders.joblib'
# SELECTOR_PATH = MODEL_DIR / 'feature_selector.joblib' # REMOVE - No longer needed
INVERSE_MAPS_PATH = MODEL_DIR / 'inverse_maps.joblib'
FEATURE_COLS_PATH = MODEL_DIR / 'feature_cols.joblib' # Still needed for column order

# --- Environment/Secrets (Keep as is, though GDrive ID for selector is now irrelevant) ---
# ... (keep environment detection and secrets handling if needed for other purposes) ...
# SELECTOR_GDRIVE_ID = "" # No longer needed

# --- Helper Functions (Remove download_from_gdrive if not used elsewhere) ---
# def download_from_gdrive(...): REMOVE if selector download was its only use

# --- REMOVE SimpleSelector Class ---
# class SimpleSelector: REMOVE

# --- Load Artifacts ---
@st.cache_resource(show_spinner="Loading model artifacts...")
def load_model_artifacts():
    """Load model artifacts (model trained on all features)."""
    artifacts = {}
    loaded_status = {}

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Define artifacts to load (NO SELECTOR)
    artifact_paths = {
        "model": MODEL_PATH,
        "label_encoders": ENCODERS_PATH,
        "inverse_maps": INVERSE_MAPS_PATH,
        "feature_cols": FEATURE_COLS_PATH,
    }

    st.info(f"Attempting to load model: {MODEL_PATH}") # Inform which model is loading

    # Load artifacts
    for name, path in artifact_paths.items():
        if path.exists():
            try:
                artifacts[name] = joblib.load(path)
                loaded_status[name] = True
                st.success(f"✅ Loaded: {path.name}")
            except Exception as e:
                st.error(f"⚠️ Error loading {name} from {path}: {e}")
                loaded_status[name] = False
        else:
            st.error(f"❌ Artifact file not found: {path}")
            loaded_status[name] = False

    # Final check for required artifacts
    required_artifacts = ["model", "label_encoders", "inverse_maps", "feature_cols"]
    missing_required = [name for name in required_artifacts if not loaded_status.get(name, False)]

    if missing_required:
        st.error(f"❌ Failed to load required artifacts: {', '.join(missing_required)}. Cannot proceed.")
        st.stop()

    # Return only the loaded artifacts (no selector, no fallback flag)
    return artifacts

# --- Load Model Artifacts ---
try:
    loaded_artifacts = load_model_artifacts()
    model = loaded_artifacts.get("model")
    label_encoders = loaded_artifacts.get("label_encoders")
    # selector = None # Explicitly set to None or remove usage
    inverse_maps = loaded_artifacts.get("inverse_maps")
    original_feature_cols = loaded_artifacts.get("feature_cols")
    st.success("✅ Models and necessary artifacts loaded successfully.")
except Exception as e:
    st.error(f"A critical error occurred during artifact loading: {e}")
    st.stop()

# --- Sidebar Navigation (Keep as is) ---
st.sidebar.title("Navigation")
pages = ["Predictions", "Data Overview (EDA)", "Model Explainability"]
selected_page = st.sidebar.radio("Go to", pages)

# --- Page Content ---

# == Predictions Page ==
if selected_page == "Predictions":
    st.title("📊 Demographic Predictor (All Features)") # Update title
    st.markdown("""
        Enter spending habit indicators below to predict user demographics based on *all* input features.
    """)

    # Define Form Fields and Options (Keep as is)
    # ... (form_fields and dropdown_options definitions remain the same) ...
    form_fields = [
        {"name": "Period", "label": "Period (YYYY-MM)", "alias": "Period"},
        {"name": "Location", "label": "Location", "alias": "Location"},
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
        {"name": "Clothing", "label": "Clothing", "alias": "Clothing"},
        {"name": "Airtime_Data_bundles", "label": "Airtime/Data", "alias": "Airtime/Data bundles"}
    ]

    spending_options = ["More", "Less", "Same", "Not sure"]
    dropdown_options = {
        field["name"]: spending_options for field in form_fields
        if field["name"] not in ["Period", "Location"]
    }
    dropdown_options["Period"] = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"]
    dropdown_options["Location"] = ["MS1", "MS2", "MS3", "MS4", "MS5"]

    # --- Create Input Form (Keep as is) ---
    with st.form("prediction_form"):
        st.subheader("Input Features")
        num_columns = 3
        cols = st.columns(num_columns)
        for i, field in enumerate(form_fields):
            col_index = i % num_columns
            with cols[col_index]:
                st.selectbox(
                    label=field["label"],
                    options=dropdown_options[field["name"]],
                    key=field["name"],
                    index=0
                )
        submit_button = st.form_submit_button("✨ Predict Demographics")

    # --- Process Prediction ---
    if submit_button:
        try:
            st.markdown("---")
            st.subheader("Processing Prediction...")

            # 1. Collect input data
            input_data = {}
            for field in form_fields:
                alias = field.get("alias", field["name"])
                input_data[alias] = st.session_state[field["name"]]
            st.write("Collected Input:")
            st.json(input_data)

            # 2. Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # 3. Ensure correct column order
            if not original_feature_cols:
                 st.error("Original feature column list not loaded.")
                 st.stop()
            try:
                input_df = input_df[original_feature_cols]
            except KeyError as e:
                st.error(f"Input data is missing expected columns: {e}.")
                st.stop()
            st.write("Input DataFrame (Ordered):")
            st.dataframe(input_df)

            # 4. Apply Label Encoding
            input_encoded = input_df.copy()
            st.write("Applying Label Encoders...")
            # (Keep the existing encoding loop - it's still needed)
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
                             st.warning(f"Unseen value '{item}' in column '{col}'. Mapping to {unknown_marker}.")
                             encoded_values.append(unknown_marker)
                    input_encoded[col] = encoded_values
                else:
                     st.warning(f"Column '{col}' from LE not found.")
            st.write("Encoded DataFrame (Input to Model):")
            st.dataframe(input_encoded) # This is now the direct input to the model

            # 5. SKIP Feature Selection Step
            st.write("Skipping feature selection step.")
            # input_selected = selector.transform(input_encoded) # REMOVE THIS LINE
            # Use the fully encoded data directly
            input_for_model = input_encoded

            # 5.1 Convert to NumPy array to avoid potential feature name mismatch warnings
            if isinstance(input_for_model, pd.DataFrame):
                st.write("Converting encoded DataFrame to NumPy array for prediction.")
                input_for_model_np = input_for_model.values
            else:
                input_for_model_np = input_for_model # Assume it's already compatible if not DataFrame

            st.write(f"Shape of data passed to model: {input_for_model_np.shape}")

            # 6. Make Prediction & Get Probabilities
            st.write("Making Prediction...")
            # (The rest of the predict_proba/predict logic and results display remains the same)
            # --- Start of unmodified prediction logic ---
            if hasattr(model, "predict_proba"):
                try:
                    # USE THE FULLY ENCODED NUMPY ARRAY
                    prediction_probas_list = model.predict_proba(input_for_model_np)
                    # st.write(prediction_probas_list) # DEBUG

                    results = {}
                    expected_target_order = ['Gender', 'Age-group', 'Occupation', 'Income Level']

                    if len(prediction_probas_list) != len(expected_target_order):
                         st.error(f"Probability/target mismatch.")
                         st.stop()

                    for i, target_name in enumerate(expected_target_order):
                        # (Keep the existing probability processing loop)
                        probas = prediction_probas_list[i][0]
                        predicted_index = np.argmax(probas)
                        predicted_proba = probas[predicted_index]
                        try:
                            base_estimator = model.estimators_[i]
                            if hasattr(base_estimator, 'best_estimator_'):
                                final_estimator = base_estimator.best_estimator_
                            else:
                                final_estimator = base_estimator
                            if not hasattr(final_estimator, 'classes_'):
                                raise AttributeError("Estimator missing 'classes_'")
                            classes = final_estimator.classes_
                            predicted_encoded_label = classes[predicted_index]
                            inv_map = inverse_maps.get(target_name)
                            if inv_map:
                                predicted_decoded_label = inv_map.get(predicted_encoded_label, f"UnkCode({predicted_encoded_label})")
                            else:
                                predicted_decoded_label = f"NoMap({predicted_encoded_label})"
                            results[target_name] = {"prediction": predicted_decoded_label, "confidence": predicted_proba}
                        except Exception as e:
                             st.error(f"Error processing prob for '{target_name}': {e}")
                             results[target_name] = {"prediction": "Error", "confidence": 0.0}

                except Exception as e: # Catch errors during predict_proba itself
                    st.warning(f"predict_proba failed: {e}. Falling back to predict().")
                    # Fallback logic using predict() - keep as is
                    prediction_encoded = model.predict(input_for_model_np)
                    # ... (rest of fallback logic) ...
                    prediction_values = prediction_encoded[0]
                    results = {}
                    expected_target_order = ['Gender', 'Age-group', 'Occupation', 'Income Level']
                    for i, target_name in enumerate(expected_target_order):
                        encoded_val = prediction_values[i]
                        inv_map = inverse_maps.get(target_name)
                        decoded_val = inv_map.get(encoded_val, f"Unknown code ({encoded_val})")
                        results[target_name] = {"prediction": decoded_val, "confidence": None}


            else: # Fallback if predict_proba doesn't exist
                st.warning("Model lacks predict_proba. Showing predictions without confidence.")
                # Fallback logic using predict() - keep as is
                prediction_encoded = model.predict(input_for_model_np)
                # ... (rest of fallback logic) ...
                prediction_values = prediction_encoded[0]
                results = {}
                expected_target_order = ['Gender', 'Age-group', 'Occupation', 'Income Level']
                for i, target_name in enumerate(expected_target_order):
                    encoded_val = prediction_values[i]
                    inv_map = inverse_maps.get(target_name)
                    decoded_val = inv_map.get(encoded_val, f"Unknown code ({encoded_val})")
                    results[target_name] = {"prediction": decoded_val, "confidence": None}

            # --- End of unmodified prediction logic ---

            # 7. Display Results (Keep as is)
            st.markdown("---")
            st.subheader("✅ Prediction Results")
            res_cols = st.columns(len(results))
            for idx, (target, value_dict) in enumerate(results.items()):
                 with res_cols[idx]:
                     pred_label = value_dict['prediction']
                     confidence = value_dict['confidence']
                     if confidence is not None:
                         st.metric(label=target, value=pred_label, delta=f"{confidence:.1%} confidence", delta_color="off")
                     else:
                         st.metric(label=target, value=pred_label, delta="Confidence N/A", delta_color="off")
            st.markdown("---")

        except Exception as e:
            st.error(f"An error occurred during prediction processing: {str(e)}")
            st.error(traceback.format_exc())

# == Data Overview (EDA) Page (Keep as is) ==
elif selected_page == "Data Overview (EDA)":
    # ... (Keep your EDA section code) ...
    st.title("📈 Data Overview (EDA)")
    st.write("This section provides a brief overview and exploratory data analysis of the dataset used for training.")
    st.write("*(Placeholder for EDA content)*")
    try:
        data_path = Path("data.csv")
        if data_path.exists():
            df_eda = pd.read_csv(data_path)
            st.subheader("Sample Data")
            st.dataframe(df_eda.head())
            st.subheader("Data Shape")
            st.write(f"Rows: {df_eda.shape[0]}, Columns: {df_eda.shape[1]}")
            st.subheader("Basic Statistics (Numerical)")
            st.dataframe(df_eda.describe())
            st.subheader("Value Counts (Example: Age Group)")
            if 'Age-group' in df_eda.columns:
                st.bar_chart(df_eda['Age-group'].value_counts())
        else:
            st.warning("Could not find data.csv for EDA.")
    except Exception as e:
        st.error(f"Error loading or processing data for EDA: {e}")


# == Model Explainability Page (Adjust for all features) ==
elif selected_page == "Model Explainability":
    st.title("🧠 Model Explainability (All Features)")
    st.write("Insights into how the model makes predictions using all input features.")
    st.write("*(Placeholder for explainability content)*")

    st.subheader("Feature Importances (Example)")
    st.write("Relative contribution of each feature. Based on the model trained with all features.")

    try:
        importances_available = False
        feature_importances_dict = {}

        if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
            # Now, feature names directly correspond to original_feature_cols
            # as selection was skipped.
            feature_names = original_feature_cols # Use the full list

            for i, target_name in enumerate(['Gender', 'Age-group', 'Occupation', 'Income Level']):
                base_estimator = model.estimators_[i]
                final_estimator = base_estimator
                if hasattr(base_estimator, 'best_estimator_'): # Handle GridSearchCV
                    final_estimator = base_estimator.best_estimator_

                if hasattr(final_estimator, 'feature_importances_'):
                    importances = final_estimator.feature_importances_
                    # Ensure the number of importances matches the number of original features
                    if len(feature_names) == len(importances):
                        feature_importances_dict[target_name] = pd.Series(importances, index=feature_names)
                        importances_available = True
                    else:
                         st.warning(f"Mismatch: {len(feature_names)} features vs {len(importances)} importances for target '{target_name}'.")
                else:
                    st.info(f"Final estimator for '{target_name}' does not have feature_importances_.")


        if importances_available:
            for target_name, importance_series in feature_importances_dict.items():
                st.markdown(f"**Importances for predicting: {target_name}**")
                # Sort and display
                importance_df = importance_series.sort_values(ascending=False).reset_index()
                importance_df.columns = ['Feature', 'Importance']
                st.dataframe(importance_df)
                # Consider showing only top N features if the list is long
                # st.bar_chart(importance_df.head(10).set_index('Feature'))
        else:
            st.info("Feature importances are not readily available for this model configuration.")

    except Exception as e:
        st.error(f"Error trying to display feature importances: {e}")
        