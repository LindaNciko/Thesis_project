# ========== app.py (Full Updated Code) ==========
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import time
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

# --- Initialize Streamlit App ---
st.set_page_config(
    page_title="Demographic Predictor", # Simplified title for header
    page_icon="📊",
    layout="wide"
)

# --- Configuration ---
MODEL_DIR = Path("model")
# === ENSURE THESE PATHS ARE CORRECT ===
MODEL_PATH = MODEL_DIR / 'multi_output_model_all_features.joblib' # Using the all-features model
ENCODERS_PATH = MODEL_DIR / 'label_encoders.joblib'
INVERSE_MAPS_PATH = MODEL_DIR / 'inverse_maps.joblib'
FEATURE_COLS_PATH = MODEL_DIR / 'feature_cols.joblib' # Still needed for column order
DATA_PATH = Path("data.csv") # Define path to the raw data file for EDA

# --- Apply Custom CSS ---
# Colors (match or complement the config.toml)
HEADER_COLOR = "#1E8449" # Darker Green for main titles/headers
SUBHEADER_COLOR = "#D5F5E3" # Lighter Green for sub-section titles
TEXT_ON_HEADER = "#FFFFFF" # White text on dark header
TEXT_ON_SUBHEADER = "#1E8449" # Dark green text on light subheader

# Apply custom CSS styles to the Streamlit app
st.markdown(f"""
<style>
    /* === Base Theme Enhancement === */
    [data-testid="stSidebar"] > div:first-child {{
        background-color: {st.get_option("theme.secondaryBackgroundColor")};
        padding-top: 1rem;
    }}

    /* === Custom Header Styling (Applied to st.title -> h1) === */
    h1 {{
        background-color: {HEADER_COLOR};
        color: {TEXT_ON_HEADER};
        padding: 1rem 1.5rem;
        border-radius: 7px;
        text-align: center;
        font-weight: bold;
        margin-top: 0px; /* Adjust if needed */
        margin-bottom: 2rem;
    }}

    /* === Custom Sub-Header Styling (Applied to st.subheader -> h2) === */
    h2 {{
        background-color: {SUBHEADER_COLOR};
        color: {TEXT_ON_SUBHEADER};
        padding: 0.6rem 1rem;
        border-radius: 5px;
        display: inline-block;
        font-weight: bold;
        margin-top: 2.5rem;
        margin-bottom: 1rem;
    }}

    /* Style the forms */
    div[data-testid="stForm"] {{
        border: 1px solid #e0e0e0;
        border-radius: 7px;
        padding: 1.5rem;
        background-color: #fdfdfd;
        margin-bottom: 1.5rem;
    }}

    /* Style the prediction result metrics */
    div[data-testid="stMetric"] {{
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
    }}
    div[data-testid="stMetric"] > label {{ /* Metric label */
        font-weight: bold;
        color: {HEADER_COLOR};
    }}
     div[data-testid="stMetric"] > div {{ /* Metric value */
        font-size: 1.5rem;
    }}

    /* Style the submit button */
    div[data-testid="stFormSubmitButton"] button {{
         width: 100%;
         background-color: {st.get_option("theme.primaryColor")};
         color: white;
         font-weight: bold;
    }}
     div[data-testid="stFormSubmitButton"] button:hover {{
         background-color: {HEADER_COLOR};
         color: white;
         border: 1px solid {HEADER_COLOR};
     }}

    /* Adjust main container padding */
    .main .block-container {{
        padding-top: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
        padding-bottom: 2rem;
    }}

    /* --- HIDE STREAMLIT DEFAULTS --- */
     #MainMenu {{visibility: hidden;}}
     footer {{visibility: hidden;}}
     header[data-testid="stHeader"] {{
        background: none;
        height: 0px;
        visibility: hidden;
     }}
</style>
""", unsafe_allow_html=True)


# --- Load Artifacts (Corrected Function) ---
@st.cache_resource(show_spinner="Loading model artifacts...")
def load_model_artifacts():
    """Load model artifacts (model trained on all features)."""
    artifacts = {}
    loaded_status = {}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Define artifacts to load (NO SELECTOR HERE) <--- CORRECTED
    artifact_paths = {
        "model": MODEL_PATH,
        "label_encoders": ENCODERS_PATH,
        "inverse_maps": INVERSE_MAPS_PATH,
        "feature_cols": FEATURE_COLS_PATH,
    }

    # Load artifacts
    for name, path in artifact_paths.items():
        print(f"Attempting to load: {path}") # Print path being tried
        if path.exists():
            try:
                artifacts[name] = joblib.load(path)
                loaded_status[name] = True
                print(f"✅ Successfully loaded: {path.name}")
            except Exception as e:
                print(f"⚠️ Error loading {name} from {path}: {e}") # Log error to console
                loaded_status[name] = False
        else:
            print(f"❌ Artifact file not found: {path}") # Log error to console
            loaded_status[name] = False

    # Final check for required artifacts
    required_artifacts = ["model", "label_encoders", "inverse_maps", "feature_cols"]
    missing_required = [name for name in required_artifacts if not loaded_status.get(name, False)]

    if missing_required:
        # Display error prominently in the UI if loading fails critically
        st.error(f"❌ Failed to load required artifacts: {', '.join(missing_required)}. App cannot proceed.")
        st.stop() # Stop execution if essential files are missing

    return artifacts

# --- Load EDA Data ---
@st.cache_data(show_spinner="Loading data for EDA...")
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
    inverse_maps = loaded_artifacts.get("inverse_maps")
    original_feature_cols = loaded_artifacts.get("feature_cols")
    # Success message removed for cleaner UI
except Exception as e:
    st.error(f"A critical error occurred during artifact loading: {e}")
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
pages = {
    "Predictions": "🔮",
    "Data Overview (EDA)": "📊",
    "Model Explainability": "🧠",
    # "About": "ℹ️" # Uncomment to add an About page
}
selected_page_label = st.sidebar.radio("Go to", list(pages.keys()))
selected_page_icon = pages[selected_page_label]


# --- Page Content ---

# == Predictions Page ==
if selected_page_label == "Predictions":
    st.title(f"{selected_page_icon} Demographic Predictor")
    st.markdown("""
        Enter spending habit indicators below to predict user demographics based on *all* input features.
    """)

    # Define Form Fields and Options
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
    # --- IMPORTANT: Ensure these lists cover ALL values seen during training AND prediction ---
    # If you add new values here (e.g., new periods), you MUST RETRAIN the model
    # with data including those values and update the label_encoders.joblib file.
    dropdown_options["Period"] = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"]
    dropdown_options["Location"] = ["MS1", "MS2", "MS3", "MS4", "MS5"]

    # --- Create Input Form ---
    with st.form("prediction_form"):
        st.subheader("Input Features") # Styled subheader
        num_columns = 3
        cols = st.columns(num_columns)
        col_idx = 0
        fields_per_col = (len(form_fields) + num_columns - 1) // num_columns
        for i, field in enumerate(form_fields):
            current_col = cols[col_idx]
            with current_col:
                st.selectbox(
                    label=field["label"],
                    options=dropdown_options[field["name"]],
                    key=field["name"],
                    index=0 # Default selection
                )
            # Move to next column after filling allocated fields
            if (i + 1) % fields_per_col == 0 and col_idx < num_columns - 1:
                col_idx += 1

        submit_button = st.form_submit_button("✨ Predict Demographics")

    # --- Process Prediction ---
    if submit_button:
        with st.spinner("Processing Prediction..."):
            try:
                # 1. Collect input data
                input_data = {}
                for field in form_fields:
                    alias = field.get("alias", field["name"])
                    input_data[alias] = st.session_state[field["name"]]

                # 2. Convert to DataFrame
                input_df = pd.DataFrame([input_data])

                # 3. Ensure correct column order
                if not original_feature_cols:
                    st.error("Original feature column list not loaded.")
                    st.stop()
                try:
                    input_df = input_df.reindex(columns=original_feature_cols)
                except KeyError as e:
                    st.error(f"Input data is missing expected columns or order is incorrect: {e}.")
                    st.stop()

                # 4. Apply Label Encoding (Handle unseen values silently or log)
                input_encoded = input_df.copy()
                encoding_issues = []
                for col, le in label_encoders.items():
                    if col in input_encoded.columns:
                        current_col_data = input_encoded[col].astype(str)
                        known_classes = set(le.classes_)
                        unknown_marker = -1 # Define how to handle unseen
                        encoded_values = []
                        for item in current_col_data:
                            if item in known_classes:
                                encoded_values.append(le.transform([item])[0])
                            else:
                                # Log unseen values instead of showing warnings in UI
                                encoding_issues.append(f"Unseen value '{item}' in column '{col}'. Mapped to {unknown_marker}.")
                                encoded_values.append(unknown_marker)
                        input_encoded[col] = encoded_values

                if encoding_issues:
                     print("Encoding Issues Encountered During Prediction:")
                     for issue in encoding_issues:
                         print(f"- {issue}")
                     # Optionally show a single non-spammy warning:
                     # st.warning("Some input values were not seen during training. Results may be less accurate.")

                # 5. Skip Feature Selection (Already handled by using correct model)
                input_for_model = input_encoded

                # 5.1 Convert to NumPy
                if isinstance(input_for_model, pd.DataFrame):
                    input_for_model_np = input_for_model.values
                else:
                    input_for_model_np = input_for_model

                # 6. Make Prediction & Get Probabilities
                results = {}
                expected_target_order = ['Gender', 'Age-group', 'Occupation', 'Income Level']

                if hasattr(model, "predict_proba"):
                    try:
                        prediction_probas_list = model.predict_proba(input_for_model_np)
                        if len(prediction_probas_list) != len(expected_target_order):
                            st.error(f"Probability/target mismatch.")
                            st.stop()

                        for i, target_name in enumerate(expected_target_order):
                            probas = prediction_probas_list[i][0]
                            predicted_index = np.argmax(probas)
                            predicted_proba = probas[predicted_index]
                            try:
                                base_estimator = model.estimators_[i]
                                final_estimator = getattr(base_estimator, 'best_estimator_', base_estimator)

                                if not hasattr(final_estimator, 'classes_'):
                                    raise AttributeError(f"Estimator for '{target_name}' missing 'classes_'.")
                                classes = final_estimator.classes_
                                if predicted_index >= len(classes):
                                     raise IndexError(f"Predicted index out of bounds for '{target_name}'.")

                                predicted_encoded_label = classes[predicted_index]
                                inv_map = inverse_maps.get(target_name)
                                predicted_decoded_label = inv_map.get(predicted_encoded_label, f"UnkCode({predicted_encoded_label})") if inv_map else f"NoMap({predicted_encoded_label})"
                                results[target_name] = {"prediction": predicted_decoded_label, "confidence": predicted_proba}
                            except Exception as inner_e:
                                st.error(f"Error processing probabilities for '{target_name}': {inner_e}")
                                results[target_name] = {"prediction": "Error", "confidence": 0.0}

                    except Exception as e:
                        st.warning(f"predict_proba failed: {e}. Falling back to predict().")
                        # Fallback logic
                        prediction_encoded = model.predict(input_for_model_np)
                        prediction_values = prediction_encoded[0]
                        if len(prediction_values) != len(expected_target_order):
                            st.error(f"Fallback Prediction/target mismatch.")
                            st.stop()
                        for i, target_name in enumerate(expected_target_order):
                            encoded_val = prediction_values[i]
                            inv_map = inverse_maps.get(target_name)
                            decoded_val = inv_map.get(encoded_val, f"UnkCode({encoded_val})") if inv_map else f"NoMap({encoded_val})"
                            results[target_name] = {"prediction": decoded_val, "confidence": None}
                else:
                     # Fallback if predict_proba doesn't exist
                    st.warning("Model lacks predict_proba. Showing predictions without confidence.")
                    prediction_encoded = model.predict(input_for_model_np)
                    prediction_values = prediction_encoded[0]
                    if len(prediction_values) != len(expected_target_order):
                        st.error(f"Fallback Prediction/target mismatch.")
                        st.stop()
                    for i, target_name in enumerate(expected_target_order):
                        encoded_val = prediction_values[i]
                        inv_map = inverse_maps.get(target_name)
                        decoded_val = inv_map.get(encoded_val, f"UnkCode({encoded_val})") if inv_map else f"NoMap({encoded_val})"
                        results[target_name] = {"prediction": decoded_val, "confidence": None}


                # 7. Display Results
                st.subheader("✅ Prediction Results") # Styled subheader
                res_cols = st.columns(len(results))
                for idx, (target, value_dict) in enumerate(results.items()):
                    with res_cols[idx]:
                        pred_label = value_dict['prediction']
                        confidence = value_dict['confidence']
                        delta_text = f"{confidence:.1%} confidence" if confidence is not None else "Confidence N/A"
                        st.metric(label=target, value=pred_label, delta=delta_text, delta_color="off")

            except Exception as e:
                st.error(f"An unexpected error occurred during prediction processing: {str(e)}")
                st.error(traceback.format_exc())


# == Data Overview (EDA) Page ==
elif selected_page_label == "Data Overview (EDA)":
    st.title(f"{selected_page_icon} Data Overview (EDA)")
    st.markdown("Exploratory data analysis of the raw dataset used for training.")

    df_eda = load_eda_data(DATA_PATH)

    if df_eda is not None:
        with st.expander("DEBUG: Column Names in data.csv (Verify & Update Code!)", expanded=False):
             st.write("**ACTION REQUIRED:** Inspect these names carefully and update the hardcoded names (marked with <<< TODO >>>) in the code below if they don't match your `data.csv` file.")
             actual_columns = df_eda.columns.tolist()
             st.write(actual_columns)

        st.subheader("Sample Data")
        st.dataframe(df_eda.head())

        st.subheader("Data Dimensions")
        col_dim1, col_dim2 = st.columns(2)
        col_dim1.metric("Number of Rows", df_eda.shape[0])
        col_dim2.metric("Number of Columns", df_eda.shape[1])

        # --- Visualizations ---
        st.markdown("---")
        st.subheader("Distributions of Key Demographics")

        # <<< TODO: VERIFY/UPDATE these hardcoded names based on your actual data.csv column names >>>
        eda_gender_col = 'Gender'
        eda_income_col = 'Income Level'
        eda_age_col = 'Age-group'
        eda_occupation_col = 'Occupation'

        col1, col2 = st.columns(2)
        # Function to plot bar chart safely
        def plot_bar(df, col_name, display_name, target_col, order=None):
             with target_col:
                 st.markdown(f"**{display_name}**")
                 if col_name in df.columns:
                     counts = df[col_name].value_counts()
                     if order:
                         try:
                             counts = counts.reindex(order, fill_value=0)
                         except Exception: pass # Ignore reindex errors
                     st.bar_chart(counts)
                 else: st.warning(f"Column '{col_name}' not found.")

        plot_bar(df_eda, eda_gender_col, "Gender Distribution", col1)
        plot_bar(df_eda, eda_income_col, "Income Level Distribution", col1, ['Low Income', 'Middle Income', 'High Income']) # Adjust order if needed
        plot_bar(df_eda, eda_age_col, "Age Group Distribution", col2, ['Baby Boomers', 'Gen X', 'Millennials', 'Gen Z']) # Adjust order if needed
        plot_bar(df_eda, eda_occupation_col, "Occupation Distribution", col2)


        st.markdown("---")
        st.subheader("Spending Habit Examples")
        # <<< TODO: VERIFY/UPDATE these hardcoded names based on your actual data.csv column names >>>
        spending_cols_to_plot = [
            'Personal Hygiene (eg Soap, toothpaste)', 'Fresh groceries (Fruits, vegetables)',
            'Entertainment (eg. Restaurants, movies)', 'Electronics (eg Phone, Computers)',
            'Alcohol beverages', 'Clothing'
        ]
        valid_spending_cols = [col for col in spending_cols_to_plot if col in actual_columns]
        missing_spending_cols = [col for col in spending_cols_to_plot if col not in actual_columns]

        if not valid_spending_cols:
             st.warning("None of the example spending habit columns were found in data.csv.")
        else:
            num_spend_cols = min(len(valid_spending_cols), 3)
            cols_spend = st.columns(num_spend_cols)
            spend_order = ['More', 'Same', 'Less', 'Not sure'] # Adjust order if needed
            for i, col_name in enumerate(valid_spending_cols):
                 display_label = col_name.split('(')[0].strip() if '(' in col_name else col_name
                 plot_bar(df_eda, col_name, display_label, cols_spend[i % num_spend_cols], spend_order)

        if missing_spending_cols:
             st.warning(f"Could not plot distributions for some spending habits (columns not found): {', '.join(missing_spending_cols)}")


        st.markdown("---")
        st.subheader("Relationship Plots (Heatmaps)")
        # <<< TODO: VERIFY/UPDATE these hardcoded names >>>
        eda_rel_age_col = 'Age-group'
        eda_rel_income_col = 'Income Level'
        eda_rel_gender_col = 'Gender'
        eda_rel_occupation_col = 'Occupation'

        # Function to plot heatmap safely
        def plot_heatmap(df, index_col, columns_col, title, target_elem, index_order=None, columns_order=None, cmap="Blues"):
            target_elem.markdown(f"**{title}**")
            if index_col in df.columns and columns_col in df.columns:
                try:
                    ct = pd.crosstab(df[index_col], df[columns_col])
                    ct_norm = ct.apply(lambda r: r/r.sum()*100 if r.sum() > 0 else r, axis=1)
                    if index_order or columns_order:
                        try:
                            ct_norm = ct_norm.reindex(index=index_order, columns=columns_order, fill_value=0)
                        except Exception as reindex_err:
                            target_elem.info(f"Note: Could not reindex heatmap ({reindex_err}).")

                    fig_height = max(5, len(ct_norm) * 0.4) # Dynamic height based on index categories
                    fig, ax = plt.subplots(figsize=(10, fig_height))
                    sns.heatmap(ct_norm, annot=True, fmt=".1f", cmap=cmap, ax=ax, linewidths=.5)
                    ax.set_ylabel(index_col)
                    ax.set_xlabel(columns_col)
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    plt.tight_layout()
                    target_elem.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    target_elem.error(f"Error creating '{title}' heatmap: {e}")
            else:
                target_elem.warning(f"Required columns ('{index_col}', '{columns_col}') not found for '{title}'.")

        # Plot the heatmaps
        rel_cols = st.columns(1) # Place heatmaps one below other
        # <<< TODO: Adjust order lists if your categories differ >>>
        age_order_hm = ['Baby Boomers', 'Gen X', 'Millennials', 'Gen Z']
        income_order_hm = ['Low Income', 'Middle Income', 'High Income']
        plot_heatmap(df_eda, eda_rel_age_col, eda_rel_income_col,
                     f"{eda_rel_income_col} % within {eda_rel_age_col}",
                     rel_cols[0], index_order=age_order_hm, columns_order=income_order_hm, cmap="Greens")

        st.markdown("<br>", unsafe_allow_html=True) # Add space

        plot_heatmap(df_eda, eda_rel_occupation_col, eda_rel_gender_col,
                     f"{eda_rel_gender_col} % within {eda_rel_occupation_col}",
                     rel_cols[0], cmap="Pastel1") # No specific order needed usually


    else:
        st.error("Could not load data for EDA.")


# == Model Explainability Page ==
elif selected_page_label == "Model Explainability":
    st.title(f"{selected_page_icon} Model Explainability")
    st.write("Insights into how the model makes predictions using all input features.")

    st.subheader("Global Feature Importances")
    st.write("Relative contribution of each input feature to the predictions for each demographic target, averaged across the model.")
    st.markdown("""
    *   Derived from tree-based models (e.g., Random Forest). Higher values mean greater impact.
    *   Note: Not available for all model types.
    """)

    try:
        importances_available = False
        feature_importances_dict = {}
        target_names = ['Gender', 'Age-group', 'Occupation', 'Income Level']

        if hasattr(model, 'estimators_') and len(model.estimators_) == len(target_names):
            if not original_feature_cols:
                st.error("Feature names not loaded. Cannot display importances.")
            else:
                feature_names = original_feature_cols
                for i, target_name in enumerate(target_names):
                    try:
                        base_estimator = model.estimators_[i]
                        final_estimator = getattr(base_estimator, 'best_estimator_', base_estimator)

                        if hasattr(final_estimator, 'feature_importances_'):
                            importances = final_estimator.feature_importances_
                            if len(feature_names) == len(importances):
                                feature_importances_dict[target_name] = pd.Series(importances, index=feature_names)
                                importances_available = True
                            else:
                                st.warning(f"Mismatch: {len(feature_names)} features vs {len(importances)} importances for '{target_name}'.")
                    except Exception as est_e:
                         st.warning(f"Could not access estimator or importances for '{target_name}': {est_e}")

        else:
            st.info("Feature importances require the model to have an `estimators_` attribute (common in scikit-learn multi-output wrappers with tree-based estimators).")


        if importances_available:
            num_targets_with_importance = len(feature_importances_dict)
            imp_cols = st.columns(min(num_targets_with_importance, 2))
            col_idx = 0

            for target_name, importance_series in feature_importances_dict.items():
                with imp_cols[col_idx % len(imp_cols)]:
                    st.markdown(f"**Importances for: {target_name}**")
                    importance_df = importance_series.sort_values(ascending=False).reset_index()
                    importance_df.columns = ['Feature', 'Importance']
                    # Filter out very low or zero importances for clarity
                    importance_df = importance_df[importance_df['Importance'] > 0.001]

                    n_features_to_show = 15 # Show top N features in the chart
                    if not importance_df.empty:
                        st.bar_chart(importance_df.head(n_features_to_show).set_index('Feature')['Importance'])
                        with st.expander(f"View full table for {target_name}"):
                            st.dataframe(importance_df)
                    else:
                        st.info(f"No features with importance > 0.001 found for {target_name}.")
                col_idx += 1

        elif not feature_importances_dict: # Check if the dict remained empty
             st.info("Feature importances are not available for the underlying models used in this pipeline.")

    except Exception as e:
        st.error(f"An error occurred displaying feature importances: {e}")
        st.error(traceback.format_exc())

# == About Page (Example) ==
elif selected_page_label == "About":
     st.title(f"{selected_page_icon} About")
     st.subheader("Demographic Predictor Application")
     st.markdown("""
     This application predicts user demographics (Gender, Age Group, Occupation, Income Level)
     based on their self-reported spending habit changes across various categories.

     **Data Source:**
     *   The model was trained on anonymized survey data collected [Mention Source/Period if applicable].

     **Model:**
     *   A `[Mention Model Type, e.g., MultiOutputClassifier with RandomForest]` model trained on all relevant input features.
     *   Features were label-encoded before training.

     **Purpose:**
     *   To demonstrate the potential of using spending indicators for demographic analysis.
     *   Educational tool for understanding multi-output classification and model deployment.

     **Version:** 1.1 (Themed UI, All Features Model)
     """)
     st.info("Note: Predictions are based on learned patterns and may not be perfectly accurate.")