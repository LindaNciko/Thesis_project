# ========== app.py (Modified with Visualizations) ==========
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
import matplotlib.pyplot as plt # <-- ADDED for plotting
import seaborn as sns           # <-- ADDED for plotting

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
DATA_PATH = Path("data.csv") # Define path to the raw data file for EDA

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

    # st.info(f"Attempting to load model: {MODEL_PATH}") # Inform which model is loading

    # Load artifacts
    for name, path in artifact_paths.items():
        if path.exists():
            try:
                artifacts[name] = joblib.load(path)
                loaded_status[name] = True
                # st.success(f"✅ Loaded: {path.name}")
            except Exception as e:
                # st.error(f"⚠️ Error loading {name} from {path}: {e}")
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

# --- Load EDA Data ---
@st.cache_data(show_spinner="Loading data for EDA...") # Cache data loading
def load_eda_data(data_file_path):
    """Loads the raw data CSV for EDA."""
    if not data_file_path.exists():
        st.warning(f"Could not find data file: {data_file_path}")
        return None
    try:
        df = pd.read_csv(data_file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data from {data_file_path}: {e}")
        return None

# --- Load Model Artifacts ---
try:
    loaded_artifacts = load_model_artifacts()
    model = loaded_artifacts.get("model")
    label_encoders = loaded_artifacts.get("label_encoders")
    # selector = None # Explicitly set to None or remove usage
    inverse_maps = loaded_artifacts.get("inverse_maps")
    original_feature_cols = loaded_artifacts.get("feature_cols")
    # st.success("✅ Models and necessary artifacts loaded successfully.")
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
    dropdown_options["Period"] = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"] # Example periods
    dropdown_options["Location"] = ["MS1", "MS2", "MS3", "MS4", "MS5"] # Example locations

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
                    index=0 # Default selection
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
                alias = field.get("alias", field["name"]) # Use alias if defined
                input_data[alias] = st.session_state[field["name"]]
            # st.write("Collected Input:") # Optional: Keep for debugging
            # st.json(input_data)

            # 2. Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # 3. Ensure correct column order
            if not original_feature_cols:
                 st.error("Original feature column list not loaded.")
                 st.stop()
            try:
                # Reindex to ensure all expected columns are present and in order
                input_df = input_df.reindex(columns=original_feature_cols)
            except KeyError as e:
                st.error(f"Input data is missing expected columns or order is incorrect: {e}.")
                st.stop()
            # st.write("Input DataFrame (Ordered):") # Optional: Keep for debugging
            # st.dataframe(input_df)

            # 4. Apply Label Encoding
            input_encoded = input_df.copy()
            # st.write("Applying Label Encoders...")
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
                            #  st.warning(f"Unseen value '{item}' in column '{col}'. Mapping to {unknown_marker}.")
                             encoded_values.append(unknown_marker)
                    input_encoded[col] = encoded_values
                else:
                     st.warning(f"Column '{col}' from LE not found.")
            st.write("Encoded DataFrame (Input to Model):")
            st.dataframe(input_encoded) # This is now the direct input to the model

            # 5. SKIP Feature Selection Step
            # st.write("Skipping feature selection step.") # Optional: Keep for debugging
            # input_selected = selector.transform(input_encoded) # REMOVE THIS LINE
            input_for_model = input_encoded # Use the fully encoded data directly

            # 5.1 Convert to NumPy array to avoid potential feature name mismatch warnings
            if isinstance(input_for_model, pd.DataFrame):
                # st.write("Converting encoded DataFrame to NumPy array for prediction.") # Optional: Keep for debugging
                input_for_model_np = input_for_model.values
            else:
                input_for_model_np = input_for_model # Assume it's already compatible

            # st.write(f"Shape of data passed to model: {input_for_model_np.shape}") # Optional: Keep for debugging

            # 6. Make Prediction & Get Probabilities
            # st.write("Making Prediction...") # Optional: Keep for debugging
            # (The rest of the predict_proba/predict logic and results display remains the same)
            # --- Start of unmodified prediction logic ---
            if hasattr(model, "predict_proba"):
                try:
                    prediction_probas_list = model.predict_proba(input_for_model_np)
                    results = {}
                    expected_target_order = ['Gender', 'Age-group', 'Occupation', 'Income Level']

                    if len(prediction_probas_list) != len(expected_target_order):
                         st.error(f"Probability/target mismatch: Expected {len(expected_target_order)} sets of probabilities, got {len(prediction_probas_list)}.")
                         st.stop()

                    for i, target_name in enumerate(expected_target_order):
                        probas = prediction_probas_list[i][0] # Get probabilities for the first (and only) input sample
                        predicted_index = np.argmax(probas)
                        predicted_proba = probas[predicted_index]
                        try:
                            # Correctly access the base estimator
                            base_estimator = model.estimators_[i]
                            # Check if it's a GridSearchCV object first
                            if hasattr(base_estimator, 'best_estimator_'):
                                final_estimator = base_estimator.best_estimator_
                            else: # Otherwise, assume it's the final estimator
                                final_estimator = base_estimator

                            if not hasattr(final_estimator, 'classes_'):
                                raise AttributeError(f"Estimator for '{target_name}' (type: {type(final_estimator).__name__}) is missing 'classes_' attribute.")

                            classes = final_estimator.classes_
                            if predicted_index >= len(classes):
                                raise IndexError(f"Predicted index ({predicted_index}) out of bounds for classes (length {len(classes)}) for target '{target_name}'.")

                            predicted_encoded_label = classes[predicted_index]
                            inv_map = inverse_maps.get(target_name)
                            if inv_map:
                                predicted_decoded_label = inv_map.get(predicted_encoded_label, f"UnkCode({predicted_encoded_label})")
                            else:
                                predicted_decoded_label = f"NoMap({predicted_encoded_label})"
                                st.warning(f"Inverse map not found for target '{target_name}'.")
                            results[target_name] = {"prediction": predicted_decoded_label, "confidence": predicted_proba}
                        except Exception as e:
                             st.error(f"Error processing probabilities for target '{target_name}': {e}")
                             st.error(traceback.format_exc()) # More detailed error for debugging
                             results[target_name] = {"prediction": "Error", "confidence": 0.0}

                except Exception as e: # Catch errors during predict_proba itself
                    st.warning(f"Model's predict_proba method failed: {e}. Falling back to predict().")
                    # Fallback logic using predict() - keep as is
                    prediction_encoded = model.predict(input_for_model_np)
                    prediction_values = prediction_encoded[0] # Get predictions for the first sample
                    results = {}
                    expected_target_order = ['Gender', 'Age-group', 'Occupation', 'Income Level']
                    if len(prediction_values) != len(expected_target_order):
                         st.error(f"Prediction/target mismatch: Expected {len(expected_target_order)} predictions, got {len(prediction_values)}.")
                         st.stop()

                    for i, target_name in enumerate(expected_target_order):
                        encoded_val = prediction_values[i]
                        inv_map = inverse_maps.get(target_name)
                        decoded_val = inv_map.get(encoded_val, f"Unknown code ({encoded_val})")
                        results[target_name] = {"prediction": decoded_val, "confidence": None}

            else: # Fallback if predict_proba doesn't exist at all
                st.warning("Model lacks predict_proba method. Showing predictions without confidence scores.")
                # Fallback logic using predict() - keep as is
                prediction_encoded = model.predict(input_for_model_np)
                prediction_values = prediction_encoded[0] # Get predictions for the first sample
                results = {}
                expected_target_order = ['Gender', 'Age-group', 'Occupation', 'Income Level']
                if len(prediction_values) != len(expected_target_order):
                     st.error(f"Prediction/target mismatch: Expected {len(expected_target_order)} predictions, got {len(prediction_values)}.")
                     st.stop()

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
            st.error(f"An unexpected error occurred during prediction processing: {str(e)}")
            st.error(traceback.format_exc())


# == Data Overview (EDA) Page (REPLACED with Visualization Code) ==
elif selected_page == "Data Overview (EDA)":
    st.title("📈 Data Overview (EDA)")
    st.markdown("Exploratory data analysis of the raw dataset used for training.")

    # Load the data using the cached function
    df_eda = load_eda_data(DATA_PATH)

    if df_eda is not None:
        # ========================= DEBUG =========================
        st.subheader("DEBUG: Columns Found in data.csv")
        st.write("**ACTION REQUIRED:** Inspect these names carefully and update the hardcoded names (marked with <<< TODO >>>) in the code below if they don't match your `data.csv` file.")
        actual_columns = df_eda.columns.tolist()
        st.write(actual_columns)
        st.markdown("---")
        # ======================= END DEBUG =======================

        st.subheader("Sample Data")
        st.dataframe(df_eda.head())

        st.subheader("Data Dimensions")
        st.write(f"Number of rows: {df_eda.shape[0]}")
        st.write(f"Number of columns: {df_eda.shape[1]}")

        # --- Visualizations ---
        st.markdown("---")
        st.subheader("Distributions of Key Demographics")

        # <<< TODO: Update these hardcoded names based on your actual data.csv column names >>>
        eda_gender_col = 'Gender'
        eda_income_col = 'Income Level'
        eda_age_col = 'Age-group'
        eda_occupation_col = 'Occupation'

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{eda_gender_col} Distribution**")
            if eda_gender_col in actual_columns:
                gender_counts = df_eda[eda_gender_col].value_counts()
                st.bar_chart(gender_counts)
            else: st.warning(f"Column '{eda_gender_col}' not found in data.csv.")

            st.markdown(f"**{eda_income_col} Distribution**")
            if eda_income_col in actual_columns:
                income_counts = df_eda[eda_income_col].value_counts()
                # Define expected order - adjust if your categories differ
                income_order = ['Low Income', 'Middle Income', 'High Income']
                try: # Handle cases where expected categories might be missing
                    income_counts = income_counts.reindex(income_order, fill_value=0)
                except Exception: pass # Ignore reindex error if categories mismatch significantly
                st.bar_chart(income_counts)
            else: st.warning(f"Column '{eda_income_col}' not found in data.csv.")

        with col2:
            st.markdown(f"**{eda_age_col} Distribution**")
            if eda_age_col in actual_columns:
                age_counts = df_eda[eda_age_col].value_counts()
                # Define expected order - adjust if your categories differ
                age_order = ['Baby Boomers', 'Gen X', 'Millennials', 'Gen Z']
                try:
                    age_counts = age_counts.reindex(age_order, fill_value=0)
                except Exception: pass
                st.bar_chart(age_counts)
            else: st.warning(f"Column '{eda_age_col}' not found in data.csv.")

            st.markdown(f"**{eda_occupation_col} Distribution**")
            if eda_occupation_col in actual_columns:
                occupation_counts = df_eda[eda_occupation_col].value_counts()
                st.bar_chart(occupation_counts)
            else: st.warning(f"Column '{eda_occupation_col}' not found in data.csv.")

        st.markdown("---")
        st.subheader("Spending Habit Examples")
        # <<< TODO: Update these hardcoded names based on your actual data.csv column names >>>
        # Check the DEBUG output above for the exact names in your file
        spending_cols_to_plot = [
            'Personal Hygiene (eg Soap, toothpaste)', # Example, likely matches form alias
            'Fresh groceries (Fruits, vegetables)',    # Example, likely matches form alias
            'Entertainment (eg. Restaurants, movies)', # Example, likely matches form alias
            'Electronics (eg Phone, Computers)',       # Example, likely matches form alias
            'Alcohol beverages',                       # Example, likely matches form alias
            'Clothing'                                 # Example, likely matches form alias
        ]
        # Filter to only columns actually present in the loaded data
        valid_spending_cols = [col for col in spending_cols_to_plot if col in actual_columns]

        if not valid_spending_cols:
             st.warning("None of the example spending habit columns were found in data.csv. Check the <<< TODO >>> section above and the DEBUG output.")
        else:
            cols_spend = st.columns(min(len(valid_spending_cols), 3)) # Layout in max 3 columns
            for i, col_name in enumerate(valid_spending_cols):
                 with cols_spend[i % len(cols_spend)]:
                     # Create a slightly cleaner label for the chart title
                     display_label = col_name.split('(')[0].strip() if '(' in col_name else col_name
                     st.markdown(f"**{display_label}**")
                     # No need to check 'in actual_columns' again here
                     spend_counts = df_eda[col_name].value_counts()
                     # Define expected order for spending habits
                     spend_order = ['More', 'Same', 'Less', 'Not sure']
                     try:
                         spend_counts = spend_counts.reindex(spend_order, fill_value=0)
                     except Exception: pass
                     st.bar_chart(spend_counts)

        # Report missing columns if any were intended but not found
        missing_spending_cols = [col for col in spending_cols_to_plot if col not in actual_columns]
        if missing_spending_cols:
             st.warning(f"Could not plot distributions for some spending habits (columns not found): {', '.join(missing_spending_cols)}")


        st.markdown("---")
        st.subheader("Relationship Plots (using Heatmaps)")
        # <<< TODO: Update these hardcoded names based on your actual data.csv column names >>>
        eda_rel_age_col = 'Age-group'
        eda_rel_income_col = 'Income Level'
        eda_rel_gender_col = 'Gender'
        eda_rel_occupation_col = 'Occupation'

        # --- Age Group vs Income Level Heatmap ---
        if eda_rel_age_col in actual_columns and eda_rel_income_col in actual_columns:
            st.markdown(f"**{eda_rel_income_col} Distribution within each {eda_rel_age_col} (%)**")
            try:
                ct = pd.crosstab(df_eda[eda_rel_age_col], df_eda[eda_rel_income_col])
                # Normalize by row (percentage within each age group)
                ct_norm = ct.apply(lambda r: r/r.sum()*100 if r.sum() > 0 else r, axis=1)

                # Reorder for better visualization if needed - use orders defined earlier
                age_order = ['Baby Boomers', 'Gen X', 'Millennials', 'Gen Z']
                income_order = ['Low Income', 'Middle Income', 'High Income']
                try: # Attempt reindexing, ignore if categories don't perfectly match
                    ct_norm = ct_norm.reindex(index=age_order, columns=income_order, fill_value=0)
                except Exception as reindex_err:
                     st.info(f"Note: Could not perfectly reindex Age/Income heatmap ({reindex_err}). Displaying with original order.")

                fig, ax = plt.subplots(figsize=(10, 6)) # Create figure and axes
                sns.heatmap(ct_norm, annot=True, fmt=".1f", cmap="Blues", ax=ax, linewidths=.5)
                # ax.set_title(f'{eda_rel_income_col} Distribution within each {eda_rel_age_col} (%)') # Title inside plot can be redundant
                ax.set_ylabel(eda_rel_age_col) # Set axis labels
                ax.set_xlabel(eda_rel_income_col)
                plt.xticks(rotation=45, ha='right') # Rotate x-axis labels if long
                plt.yticks(rotation=0) # Ensure y-axis labels are horizontal
                plt.tight_layout() # Adjust layout
                st.pyplot(fig) # Display the plot in Streamlit
                plt.close(fig) # Close the figure to free memory
            except Exception as e:
                st.error(f"Error creating Age/Income heatmap: {e}")
                st.error(traceback.format_exc())
        else:
            st.warning(f"Required columns ('{eda_rel_age_col}', '{eda_rel_income_col}') not found for Age/Income relationship plot.")

        # --- Occupation vs Gender Heatmap ---
        if eda_rel_occupation_col in actual_columns and eda_rel_gender_col in actual_columns:
             st.markdown(f"**{eda_rel_gender_col} Distribution within each {eda_rel_occupation_col} (%)**")
             try:
                 ct_occ_gen = pd.crosstab(df_eda[eda_rel_occupation_col], df_eda[eda_rel_gender_col])
                 ct_occ_gen_norm = ct_occ_gen.apply(lambda r: r/r.sum()*100 if r.sum() > 0 else r, axis=1)

                 # Dynamically adjust figure height based on number of occupations
                 num_occupations = len(ct_occ_gen_norm)
                 fig_height = max(5, num_occupations * 0.4) # Min height 5, scale with occupations

                 fig, ax = plt.subplots(figsize=(8, fig_height))
                 sns.heatmap(ct_occ_gen_norm, annot=True, fmt=".1f", cmap="Pastel1", ax=ax, linewidths=.5)
                 ax.set_ylabel(eda_rel_occupation_col)
                 ax.set_xlabel(eda_rel_gender_col)
                 plt.xticks(rotation=0) # Keep gender labels horizontal
                 plt.yticks(rotation=0) # Keep occupation labels horizontal
                 plt.tight_layout() # Adjust layout
                 st.pyplot(fig)
                 plt.close(fig)
             except Exception as e:
                 st.error(f"Error creating Occupation/Gender heatmap: {e}")
                 st.error(traceback.format_exc())
        else:
            st.warning(f"Required columns ('{eda_rel_occupation_col}', '{eda_rel_gender_col}') not found for Occupation/Gender relationship plot.")


    else:
        # This message is shown if load_eda_data returned None
        st.error("Could not load data for EDA. Please ensure 'data.csv' exists in the correct location and is readable.")


# == Model Explainability Page (Adjust for all features) ==
elif selected_page == "Model Explainability":
    st.title("🧠 Model Explainability (All Features)")
    st.write("Insights into how the model makes predictions using all input features.")
    # st.write("*(Placeholder for explainability content)*") # Remove placeholder

    st.subheader("Feature Importances (Global)")
    st.write("Relative contribution of each input feature to the predictions for each demographic target. Based on the model trained with **all features**.")
    st.markdown("""
    *   These importances are typically derived from tree-based models (Random Forest) within the pipeline.
    *   Higher values indicate that the feature has a greater impact on the model's decisions for that specific target variable, averaged across all the trees in the ensemble.
    *   **Note:** If the underlying model doesn't provide feature importances, this section might be empty or show a message.
    """)

    try:
        importances_available = False
        feature_importances_dict = {}
        target_names = ['Gender', 'Age-group', 'Occupation', 'Income Level'] # Consistent order

        if hasattr(model, 'estimators_') and len(model.estimators_) == len(target_names):
            # Feature names should correspond to the original_feature_cols used for training
            if not original_feature_cols:
                st.error("Feature names (original_feature_cols) not loaded. Cannot display importances correctly.")
            else:
                feature_names = original_feature_cols # Use the full list from training

                for i, target_name in enumerate(target_names):
                    base_estimator = model.estimators_[i]
                    final_estimator = base_estimator
                    # Handle potential nesting (e.g., GridSearchCV, Pipeline)
                    if hasattr(base_estimator, 'best_estimator_'): # Handle GridSearchCV
                        final_estimator = base_estimator.best_estimator_
                    # You might need to add more checks if you have deeper pipelines, e.g.:
                    # if hasattr(final_estimator, 'named_steps'):
                    #     final_estimator = final_estimator.named_steps['classifier'] # Or model step name

                    if hasattr(final_estimator, 'feature_importances_'):
                        importances = final_estimator.feature_importances_
                        # Ensure the number of importances matches the number of features used
                        if len(feature_names) == len(importances):
                            feature_importances_dict[target_name] = pd.Series(importances, index=feature_names)
                            importances_available = True
                        else:
                            st.warning(f"Mismatch: {len(feature_names)} features vs {len(importances)} importances found for target '{target_name}'. Cannot reliably display.")
                            st.warning(f"Final estimator type: {type(final_estimator).__name__}")
                    else:
                        st.info(f"The final estimator ({type(final_estimator).__name__}) for target '{target_name}' does not have a 'feature_importances_' attribute.")

        else:
            st.warning("Model structure does not seem to be the expected list of estimators (`model.estimators_`). Cannot extract feature importances.")


        if importances_available:
            # Use columns for better layout if showing multiple importance plots
            num_targets_with_importance = len(feature_importances_dict)
            importance_cols = st.columns(min(num_targets_with_importance, 2)) # Max 2 columns layout
            col_idx = 0

            for target_name, importance_series in feature_importances_dict.items():
                with importance_cols[col_idx % len(importance_cols)]:
                    st.markdown(f"**Importances for predicting: {target_name}**")
                    # Sort and prepare DataFrame
                    importance_df = importance_series.sort_values(ascending=False).reset_index()
                    importance_df.columns = ['Feature', 'Importance']
                    importance_df = importance_df[importance_df['Importance'] > 0.001] # Optional: Filter very low importances

                    # Display as a bar chart (Top N)
                    n_features_to_show = 15 # Show top 15 features
                    st.bar_chart(importance_df.head(n_features_to_show).set_index('Feature')['Importance'])

                    # Optionally display the full table in an expander
                    with st.expander(f"View full importance table for {target_name}"):
                        st.dataframe(importance_df)
                col_idx += 1 # Move to the next column

        elif not any('feature_importances_' in str(type(getattr(est, 'best_estimator_', est))) for est in getattr(model, 'estimators_', [])):
             # Check if *any* estimator likely has importances before showing the generic message
             st.info("Feature importances are not available for the underlying models used in this pipeline.")

    except Exception as e:
        st.error(f"An error occurred while trying to display feature importances: {e}")
        st.error(traceback.format_exc())