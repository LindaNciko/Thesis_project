# XGBoost Multiclassification Model and SHAP Visualization

This repository hosts a Streamlit application designed for exploring an XGBoost-based multiclassification model. The app provides users with insights into the model's predictions, feature importance, and SHAP visualizations for interpretability.

## Features

### 1. **XGBoost Multiclassification Model**
- Users can input feature values via an interactive sidebar.
- Perform predictions using a pre-trained XGBoost model.
- Outputs predictions with decoded labels for better readability.

### 2. **Feature Importance**
- Displays feature importance for each output target.
- Visualize feature importance using horizontal bar charts.

### 3. **SHAP Visualizations**
- Generate global and local interpretability visualizations.
- **Summary Plot**: Visualize the overall impact of features on predictions.
- **Force Plot**: Explain individual predictions for selected samples.

---
## Go to [Demo](https://retail-consumer-purchase-trends.streamlit.app/)
---

## Installation and Setup

### Prerequisites
Ensure the following are installed:
- Python 3.8+
- Streamlit
- XGBoost
- SHAP
- Matplotlib
- Scikit-learn
- Pandas
- NumPy
- Joblib

### Clone the Repository
```bash
$ git clone https://github.com/<your-repository>.git
$ cd <your-repository>
```

### Install Dependencies
```bash
$ pip install -r requirements.txt
```

### Directory Structure
```plaintext
.
├── models
│   ├── multi_classifier_xgb_reg.pkl  # Pre-trained XGBoost model
│   ├── feature_columns.pkl           # Feature columns
│   └── target_columns.pkl            # Target columns
├── data
│   ├── data.csv                      # Original dataset
│   ├── encoded.csv                   # Preprocessed dataset
│   └── X_test.csv                    # Test dataset
├── app.py                            # Main Streamlit app
└── requirements.txt                  # Required Python packages
```

### Running the Application
Start the Streamlit app:
```bash
$ streamlit run app.py
```

Access the app in your browser at `http://localhost:8501`.

---

## File Descriptions

### `app.py`
Main Streamlit application file. Implements:
- Page navigation for the multiclassification model, feature importance, and SHAP visualizations.
- User input processing and prediction logic.
- Visualization using Matplotlib and SHAP.

### `models/`
- **multi_classifier_xgb_reg.pkl**: Pre-trained XGBoost model for multiclass classification.
- **feature_columns.pkl**: Pickled file containing feature column names.
- **target_columns.pkl**: Pickled file containing target column names.

### `data/`
- **data.csv**: Raw dataset used for analysis.
- **encoded.csv**: Preprocessed dataset with encoded values.
- **X_test.csv**: Test dataset for model evaluation and SHAP visualizations.

### `requirements.txt`
List of Python dependencies required to run the app.

---

## User Interaction

### Input Features
Users can select feature values via the sidebar. For categorical features, predefined options are provided. For numerical features, users can input values directly.

### Outputs
#### Predictions
- Displays predictions for the selected input values.
- Outputs are decoded to their original labels for better understanding.

#### Feature Importance
- Provides bar charts of feature importance for each target variable.

#### SHAP Visualization
- **Summary Plot**: Shows the overall importance and direction of impact for each feature.
- **Force Plot**: Explains the contribution of features to a specific prediction.

---

## Development Workflow

### Model Training
1. Preprocess the dataset (`data.csv`) to encode categorical variables.
2. Split the dataset into training and test sets.
3. Train the XGBoost model using `MultiOutputClassifier`.
4. Save the trained model and column mappings using `joblib`.

### Feature Importance
Feature importances are extracted directly from the trained XGBoost estimators for each target variable.

### SHAP Integration
1. Use `shap.TreeExplainer` to calculate SHAP values.
2. Visualize global and local interpretability using `summary_plot` and `force_plot`.

---

## Contact

For inquiries or support, please reach out via:

- **Email**: [paulmwaura254@gmail.com](mailto:paulmwaura254@gmail.com)
- **LinkedIn**: [Paul Ndirangu](https://www.linkedin.com/in/paul-ndirangu/)

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.
