# ========== app.py (Enhanced EDA & Fixed SHAP) ==========
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import time
import traceback

# Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Import SHAP
import shap

# --- Initialize Streamlit App ---
st.set_page_config(
    page_title="Demographic Predictor (All Features)",
    page_icon="📊",
    layout="wide"
)

# --- Configuration ---
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / 'multi_output_model_all_features.joblib' # Using the all-features model
ENCODERS_PATH = MODEL_DIR / 'label_encoders.joblib'
INVERSE_MAPS_PATH = MODEL_DIR / 'inverse_maps.joblib'
FEATURE_COLS_PATH = MODEL_DIR / 'feature_cols.joblib' # Still needed for column order
DATA_PATH = Path("data.csv") # Path to the raw data for EDA

# --- Load Artifacts (Function remains largely the same) ---
@st.cache_resource(show_spinner="Loading model artifacts...")
def load_model_artifacts():
    """Load model artifacts (model trained on all features)."""
    artifacts = {}
    loaded_status = {}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    artifact_paths = {
        "model": MODEL_PATH,
        "label_encoders": ENCODERS_PATH,
        "inverse_maps": INVERSE_MAPS_PATH,
        "feature_cols": FEATURE_COLS_PATH,
    }
    st.info(f"Attempting to load model: {MODEL_PATH}")
    for name, path in artifact_paths.items():
        if path.exists():
            try:
                artifacts[name] = joblib.load(path)
                loaded_status[name] = True
            except Exception as e:
                st.error(f"⚠️ Error loading {name} from {path}: {e}")
                loaded_status[name] = False
        else:
            st.error(f"❌ Artifact file not found: {path}")
            loaded_status[name] = False

    required_artifacts = ["model", "label_encoders", "inverse_maps", "feature_cols"]
    missing_required = [name for name in required_artifacts if not loaded_status.get(name, False)]
    if missing_required:
        st.error(f"❌ Failed to load required artifacts: {', '.join(missing_required)}. Cannot proceed.")
        st.stop()
    st.success("✅ Core artifacts loaded.")
    return artifacts

# --- Load EDA Data (Cached) ---
@st.cache_data(show_spinner="Loading data for EDA...")
def load_eda_data(data_path):
    """Loads the raw data CSV for EDA."""
    if not data_path.exists():
        st.warning(f"Could not find data file for EDA: {data_path}")
        return None
    try:
        df = pd.read_csv(data_path)
        # Basic cleanup if needed (e.g., remove unnamed index column)
        if df.columns[0].strip() == '' or 'Unnamed: 0' in df.columns:
             df = df.drop(columns=[df.columns[0]], errors='ignore')
        return df
    except Exception as e:
        st.error(f"Error loading data from {data_path}: {e}")
        return None

# --- Preprocess data for SHAP (Cached) ---
@st.cache_data(show_spinner="Preparing data for SHAP explanation...")
def preprocess_data_for_shap(df_raw, feature_cols, _encoders):
    """Preprocesses raw data. Returns None if required columns are missing."""
    if df_raw is None:
        st.warning("Input data for SHAP preprocessing is None.")
        return None, [] # Return None and empty list

    try:
        # Check for missing columns *before* trying to select
        missing_cols_in_raw = [col for col in feature_cols if col not in df_raw.columns]
        if missing_cols_in_raw:
             # Raise error instead of just warning - prevents proceeding with bad data
             st.error(f"Raw data for SHAP is missing expected columns defined during training: {missing_cols_in_raw}")
             st.error(f"Please ensure '{DATA_PATH}' contains the correct data.")
             return None, [] # Return None and empty list

        # Proceed only if all columns are present
        X = df_raw[feature_cols].copy() # Select the expected columns
        X_encoded = X.copy()

        for col, le in _encoders.items():
            if col in X_encoded.columns: # Should always be true now if check passed
                X_encoded[col] = X_encoded[col].fillna('Unknown').astype(str)
                known_classes = set(le.classes_)
                X_encoded[col] = X_encoded[col].apply(
                    lambda item: le.transform([item])[0] if item in known_classes else -1
                )
        # Return encoded data and the list of columns used (which is feature_cols here)
        return X_encoded, feature_cols

    except KeyError as e:
        st.error(f"KeyError during SHAP preprocessing (column likely missing despite check - investigate): {e}.")
        return None, []
    except Exception as e:
        st.error(f"Unexpected error during SHAP preprocessing: {e}")
        st.error(traceback.format_exc())
        return None, []

# --- Load Model Artifacts ---
try:
    loaded_artifacts = load_model_artifacts()
    model = loaded_artifacts.get("model")
    label_encoders = loaded_artifacts.get("label_encoders")
    inverse_maps = loaded_artifacts.get("inverse_maps")
    original_feature_cols = loaded_artifacts.get("feature_cols") # This is the list model expects
except Exception as e:
    st.error(f"A critical error occurred during artifact loading: {e}")
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
pages = ["Predictions", "Data Overview (EDA)", "Model Explainability"]
selected_page = st.sidebar.radio("Go to", pages)

# --- Page Content ---

# == Predictions Page ==
if selected_page == "Predictions":
    st.title("📊 Demographic Predictor (All Features)")
    st.markdown("""
        Enter spending habit indicators below to predict user demographics based on *all* input features.
    """)

    # --- Define Form Fields and Options ---
    # Ensure 'alias' here matches the names in 'original_feature_cols' loaded from feature_cols.joblib
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
        {"name": "Electronics__eg__Phone__Computers_", "label": "Electronics", "alias": "Electronics (eg Phone, Computers)"},
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

    # --- Create Input Form ---
    with st.form("prediction_form"):
        st.subheader("Input Features")
        num_columns = 3
        cols = st.columns(num_columns)
        for i, field in enumerate(form_fields):
            col_index = i % num_columns
            with cols[col_index]:
                st.selectbox(
                    label=field["label"], options=dropdown_options[field["name"]],
                    key=field["name"], index=0)
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

            # 2. Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # 3. Ensure correct column order using loaded list
            if not original_feature_cols:
                 st.error("Original feature column list not loaded.")
                 st.stop()
            try:
                input_df = input_df[original_feature_cols] # Order based on training
            except KeyError as e:
                st.error(f"Input data generation is missing expected feature columns required by model: {e}.")
                st.stop()

            # 4. Apply Label Encoding
            input_encoded = input_df.copy()
            for col, le in label_encoders.items():
                if col in input_encoded.columns:
                    current_col_data = input_encoded[col].astype(str)
                    known_classes = set(le.classes_)
                    unknown_marker = -1
                    encoded_values = []
                    for item in current_col_data:
                         if item in known_classes: encoded_values.append(le.transform([item])[0])
                         else: encoded_values.append(unknown_marker) # Handle unknowns
                    input_encoded[col] = encoded_values

            # 5. Convert to NumPy (no feature selection)
            input_for_model = input_encoded
            if isinstance(input_for_model, pd.DataFrame):
                input_for_model_np = input_for_model.values
            else: input_for_model_np = input_for_model

            # 6. Make Prediction & Get Probabilities
            results = {}
            expected_target_order = ['Gender', 'Age-group', 'Occupation', 'Income Level']

            if hasattr(model, "predict_proba"):
                try:
                    prediction_probas_list = model.predict_proba(input_for_model_np)
                    if len(prediction_probas_list) != len(expected_target_order):
                         st.error(f"Probability array count ({len(prediction_probas_list)}) doesn't match expected target count ({len(expected_target_order)}).")
                         st.stop()

                    for i, target_name in enumerate(expected_target_order):
                        try:
                            probas = prediction_probas_list[i][0]
                            predicted_index = np.argmax(probas)
                            predicted_proba = probas[predicted_index]
                            base_estimator = model.estimators_[i]
                            final_estimator = base_estimator.best_estimator_ if hasattr(base_estimator, 'best_estimator_') else base_estimator
                            if not hasattr(final_estimator, 'classes_'): raise AttributeError("Estimator missing 'classes_'")
                            classes = final_estimator.classes_
                            if predicted_index >= len(classes): raise IndexError(f"Index {predicted_index} out of bounds for {len(classes)} classes")
                            predicted_encoded_label = classes[predicted_index]
                            inv_map = inverse_maps.get(target_name)
                            predicted_decoded_label = inv_map.get(predicted_encoded_label, f"UnkCode({predicted_encoded_label})") if inv_map else f"NoMap({predicted_encoded_label})"
                            results[target_name] = {"prediction": predicted_decoded_label, "confidence": predicted_proba}
                        except IndexError as ie:
                             st.error(f"Index error processing prob for '{target_name}': {ie}")
                             results[target_name] = {"prediction": "Error", "confidence": 0.0}
                        except Exception as e:
                             st.error(f"Error processing prob for '{target_name}': {e}")
                             results[target_name] = {"prediction": "Error", "confidence": 0.0}
                except Exception as e:
                    st.warning(f"predict_proba failed: {e}. Falling back.")
                    prediction_encoded = model.predict(input_for_model_np)
                    prediction_values = prediction_encoded[0]
                    for i, target_name in enumerate(expected_target_order):
                        encoded_val = prediction_values[i]
                        inv_map = inverse_maps.get(target_name); decoded_val = inv_map.get(encoded_val, f"UnkCode({encoded_val})") if inv_map else f"NoMap({encoded_val})"
                        results[target_name] = {"prediction": decoded_val, "confidence": None}
            else:
                st.warning("Model lacks predict_proba.")
                prediction_encoded = model.predict(input_for_model_np)
                prediction_values = prediction_encoded[0]
                for i, target_name in enumerate(expected_target_order):
                    encoded_val = prediction_values[i]; inv_map = inverse_maps.get(target_name); decoded_val = inv_map.get(encoded_val, f"UnkCode({encoded_val})") if inv_map else f"NoMap({encoded_val})"
                    results[target_name] = {"prediction": decoded_val, "confidence": None}

            # 7. Display Results
            st.markdown("---")
            st.subheader("✅ Prediction Results")
            res_cols = st.columns(len(results))
            for idx, (target, value_dict) in enumerate(results.items()):
                 with res_cols[idx]:
                     pred_label = value_dict['prediction']; confidence = value_dict['confidence']
                     delta_str = f"{confidence:.1%} confidence" if confidence is not None else "Confidence N/A"
                     st.metric(label=target, value=pred_label, delta=delta_str, delta_color="off")
            st.markdown("---")

        except Exception as e:
            st.error(f"An error occurred during prediction processing: {str(e)}")
            st.error(traceback.format_exc())


# == Data Overview (EDA) Page ==
elif selected_page == "Data Overview (EDA)":
    st.title("📈 Data Overview (EDA)")
    st.markdown("Exploratory data analysis of the raw dataset used for training.")

    df_eda = load_eda_data(DATA_PATH) # Load the data

    if df_eda is not None:
        # ========================= DEBUG =========================
        st.subheader("DEBUG: Columns Found in data.csv")
        st.write("ACTION REQUIRED: Inspect these names carefully and update the hardcoded names (marked with <<< TODO >>>) in the code below if they don't match your `data.csv` file.")
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

        # <<< TODO: Update these hardcoded names based on your data.csv >>>
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
                income_order = ['Low Income', 'Middle Income', 'High Income'] # Adjust if categories differ
                try: # Handle cases where expected categories might be missing
                    income_counts = income_counts.reindex(income_order, fill_value=0)
                except Exception: pass # Ignore reindex error if categories mismatch significantly
                st.bar_chart(income_counts)
            else: st.warning(f"Column '{eda_income_col}' not found in data.csv.")

        with col2:
            st.markdown(f"**{eda_age_col} Distribution**")
            if eda_age_col in actual_columns:
                age_counts = df_eda[eda_age_col].value_counts()
                age_order = ['Baby Boomers', 'Gen X', 'Millennials', 'Gen Z'] # Adjust if categories differ
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
        # <<< TODO: Update these hardcoded names based on your data.csv >>>
        spending_cols_to_plot = [
            'Personal Hygiene (eg Soap, toothpaste)',
            'Fresh groceries (Fruits, vegetables)',
            'Entertainment (eg. Restaurants, movies)',
            'Electronics (eg Phone, Computers)',
            'Alcohol beverages',
            'Clothing'
        ]
        # Filter to only columns actually present in the loaded data
        valid_spending_cols = [col for col in spending_cols_to_plot if col in actual_columns]
        if not valid_spending_cols:
             st.warning("None of the example spending habit columns were found in data.csv.")
        else:
            cols_spend = st.columns(min(len(valid_spending_cols), 3)) # Layout in max 3 columns
            for i, col_name in enumerate(valid_spending_cols):
                 with cols_spend[i % len(cols_spend)]:
                     display_label = col_name.split('(')[0].strip() if '(' in col_name else col_name
                     st.markdown(f"**{display_label}**")
                     # No need to check 'in actual_columns' again here
                     spend_counts = df_eda[col_name].value_counts()
                     st.bar_chart(spend_counts)

        # Report missing columns if any were intended
        missing_spending_cols = [col for col in spending_cols_to_plot if col not in actual_columns]
        if missing_spending_cols:
             st.warning(f"Could not plot distributions for some spending habits (columns not found): {', '.join(missing_spending_cols)}")


        st.markdown("---")
        st.subheader("Relationship Plots")
        # <<< TODO: Update these hardcoded names based on your data.csv >>>
        eda_rel_age_col = 'Age-group'
        eda_rel_income_col = 'Income Level'
        eda_rel_gender_col = 'Gender'
        eda_rel_occupation_col = 'Occupation'


        if eda_rel_age_col in actual_columns and eda_rel_income_col in actual_columns:
            st.markdown(f"**{eda_rel_income_col} Distribution within each {eda_rel_age_col} (%)**")
            try:
                ct = pd.crosstab(df_eda[eda_rel_age_col], df_eda[eda_rel_income_col])
                # Normalize by row (percentage within each age group)
                ct_norm = ct.apply(lambda r: r/r.sum()*100 if r.sum() > 0 else r, axis=1)
                # Reorder for better visualization if needed
                age_order = ['Baby Boomers', 'Gen X', 'Millennials', 'Gen Z']
                income_order = ['Low Income', 'Middle Income', 'High Income']
                try:
                    ct_norm = ct_norm.reindex(index=age_order, columns=income_order, fill_value=0)
                except Exception: pass # Ignore reindex errors

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(ct_norm, annot=True, fmt=".1f", cmap="Blues", ax=ax, linewidths=.5)
                # ax.set_title(f'{eda_rel_income_col} Distribution within each {eda_rel_age_col} (%)') # Title inside plot
                ax.set_ylabel(eda_rel_age_col)
                ax.set_xlabel(eda_rel_income_col)
                st.pyplot(fig)
                plt.close(fig) # Close the figure to free memory
            except Exception as e:
                st.error(f"Error creating Age/Income heatmap: {e}")
        else:
            st.warning(f"Required columns ('{eda_rel_age_col}', '{eda_rel_income_col}') not found for relationship plot.")

        # Add another relationship plot, e.g., Occupation vs Gender
        if eda_rel_occupation_col in actual_columns and eda_rel_gender_col in actual_columns:
             st.markdown(f"**{eda_rel_gender_col} Distribution within each {eda_rel_occupation_col} (%)**")
             try:
                 ct_occ_gen = pd.crosstab(df_eda[eda_rel_occupation_col], df_eda[eda_rel_gender_col])
                 ct_occ_gen_norm = ct_occ_gen.apply(lambda r: r/r.sum()*100 if r.sum() > 0 else r, axis=1)
                 fig, ax = plt.subplots(figsize=(8, len(ct_occ_gen_norm)*0.5 + 1)) # Adjust height dynamically
                 sns.heatmap(ct_occ_gen_norm, annot=True, fmt=".1f", cmap="Pastel1", ax=ax, linewidths=.5)
                 ax.set_ylabel(eda_rel_occupation_col)
                 ax.set_xlabel(eda_rel_gender_col)
                 st.pyplot(fig)
                 plt.close(fig)
             except Exception as e:
                 st.error(f"Error creating Occupation/Gender heatmap: {e}")
        else:
            st.warning(f"Required columns ('{eda_rel_occupation_col}', '{eda_rel_gender_col}') not found for relationship plot.")


    else:
        st.error("Could not load data for EDA.")


# == Model Explainability Page ==
elif selected_page == "Model Explainability":
    st.title("🧠 Model Explainability (All Features)")
    st.markdown("""
        Understanding *why* the model makes certain predictions using SHAP (SHapley Additive exPlanations).
        SHAP values show the contribution of each feature to the prediction for each target variable.
    """)

    # --- SHAP Explanation ---
    st.header("Global Feature Importance (SHAP Summary)")
    st.markdown("""
        This plot shows the most important features for predicting each demographic category, based on a sample of the data.
        Features are ranked by the sum of their SHAP value magnitudes over all samples.
        Higher SHAP values mean the feature had a larger impact on the model's output for that prediction.
        Note: Calculation uses a sample of the data and may take some time.
    """)

    shap_button = st.button("📊 Calculate and Show SHAP Summary Plots")

    if shap_button:
        with st.spinner("Calculating SHAP values (this might take a while)..."):
            try:
                # 1. Prepare Background Data
                df_eda_shap = load_eda_data(DATA_PATH) # Reload or reuse loaded data
                if df_eda_shap is None:
                    st.error("Cannot calculate SHAP values without data.")
                    st.stop()

                # In the Model Explainability section:
                n_samples = min(1000, len(df_eda_shap)) # Try a larger sample, e.g., 1000
                background_data_raw = df_eda_shap.sample(n_samples, random_state=42) # Sample data for SHAP background

                # Preprocess using the cached function (passing actual label_encoders)
                # It now returns encoded data AND the feature columns actually used
                background_data_encoded, shap_feature_cols = preprocess_data_for_shap(
                    background_data_raw, original_feature_cols, label_encoders
                )

                if background_data_encoded is None:
                    st.error("Failed to preprocess background data for SHAP.")
                    st.stop()

                # Convert to NumPy for the explainer
                background_data_np = background_data_encoded.values

                # 2. Explain Each Target Model
                expected_target_order = ['Gender', 'Age-group', 'Occupation', 'Income Level']
                explainers = [] # Optional: store explainers if needed later

                valid_explaination_found = False
                for i, target_name in enumerate(expected_target_order):
                    st.markdown(f"--- Explaining: **{target_name}** ---")
                    try:
                        base_estimator = model.estimators_[i]
                        # Handle potential GridSearchCV wrapper
                        final_estimator = base_estimator.best_estimator_ if hasattr(base_estimator, 'best_estimator_') else base_estimator

                        # Check if compatible with TreeExplainer (more robust check)
                        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
                        from sklearn.tree import DecisionTreeClassifier
                        is_tree_model = isinstance(final_estimator, (
                            DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier,
                            GradientBoostingClassifier, HistGradientBoostingClassifier
                        ))
                        # TODO: Add checks for xgboost/lightgbm if needed

                        # In the Model Explainability section, inside the loop:
                        if is_tree_model:
                            # ===> CHANGE perturbation method <===
                            explainer = shap.TreeExplainer(final_estimator, data=background_data_np, feature_perturbation="interventional")
                            # ===> End CHANGE <===
                            explainers.append(explainer)
                            shap_values = explainer.shap_values(background_data_np) # Pass numpy array

                            st.markdown(f"**SHAP Summary Plot for {target_name}**")
                            fig, ax = plt.subplots() # Create a new figure for each plot

                            # Plotting requires feature names. Use the names corresponding to background_data_encoded/np
                            # which are returned by preprocess_data_for_shap as shap_feature_cols
                            plot_feature_names = shap_feature_cols

                            # Handle multi-class vs binary output from shap_values
                            if isinstance(shap_values, list) and len(shap_values) > 1: # Multi-class
                                class_index_to_plot = 1 # Default: impact towards class 1 (e.g., Male, Gen X, Middle Income?) - Adjust as needed!
                                if len(shap_values) > class_index_to_plot:
                                     # Pass the encoded DataFrame for better feature value display on plot axis
                                     shap.summary_plot(shap_values[class_index_to_plot], background_data_encoded, feature_names=plot_feature_names, show=False, plot_type='dot')
                                     try:
                                         class_name = final_estimator.classes_[class_index_to_plot]
                                         # Attempt to decode class name for title
                                         inv_map_class = inverse_maps.get(target_name)
                                         decoded_class_name = inv_map_class.get(class_name, class_name) if inv_map_class else class_name
                                         plt.title(f"SHAP Values Impact towards '{decoded_class_name}' ({target_name})")
                                     except Exception: # Fallback title
                                          plt.title(f"SHAP Values Impact towards Class Index {class_index_to_plot} ({target_name})")
                                else: # Fallback if class index is invalid
                                     shap.summary_plot(shap_values[0], background_data_encoded, feature_names=plot_feature_names, show=False, plot_type='dot')
                                     plt.title(f"SHAP Values Impact towards Class 0 ({target_name})")
                            else: # Binary classification or single output array
                                shap.summary_plot(shap_values, background_data_encoded, feature_names=plot_feature_names, show=False, plot_type='dot')
                                plt.title(f"SHAP Values for {target_name}")

                            st.pyplot(fig, bbox_inches='tight') # Use tight layout
                            plt.close(fig) # Close figure explicitly
                            valid_explaination_found = True
                        else:
                            st.warning(f"Skipping SHAP for '{target_name}': Final estimator type '{type(final_estimator).__name__}' not directly compatible with TreeExplainer.")
                            st.info("Consider using shap.KernelExplainer (slower) or shap.PermutationExplainer for non-tree models.")

                    except Exception as e:
                        st.error(f"Error generating SHAP explanation for '{target_name}': {e}")
                        st.error(traceback.format_exc()) # Show traceback for SHAP errors

                if not valid_explaination_found:
                    st.error("Could not generate SHAP explanations for any target model using TreeExplainer.")

            except Exception as e:
                st.error(f"An error occurred during the overall SHAP value calculation process: {e}")
                st.error(traceback.format_exc())
    else:
        st.info("Click the button above to calculate and display SHAP summary plots.")

    # Placeholder for other SHAP plots
    st.markdown("---")
    st.subheader("Further Explainability (Future Work)")
    st.markdown("""
        - **Dependence Plots:** Show how a single feature's value affects the prediction interactively.
        - **Force Plots:** Explain individual predictions for specific user inputs entered on the Prediction page.
    """)