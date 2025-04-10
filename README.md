# Demographic Predictor App

A Streamlit application for predicting demographics based on spending habits.

## Prerequisites

- Python 3.9+
- Git
- Git LFS (for handling large model files)

## Local Development

1. Install Git LFS if you haven't already:
```bash
git lfs install
```

2. Clone the repository and pull the LFS files:
```bash
git clone <repository-url>
cd <repository-directory>
git lfs pull
```

3. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Run the application:
```bash
streamlit run app.py
```

## Deployment on Render.com

1. Push your code to GitHub:
```bash
git add .
git commit -m "Prepare for Render deployment"
git push
```

2. Go to [Render.com](https://render.com) and:
   - Create a new Web Service
   - Connect your GitHub repository
   - Select the branch to deploy
   - Choose "Python" as the environment
   - The service will automatically detect the `render.yaml` configuration

3. Wait for the deployment to complete. Your app will be available at the provided Render URL.

## Model Files

The following model files are required and should be present in the `model` directory:
- `feature_selector.joblib`
- `multi_output_model.joblib`
- `label_encoders.joblib`
- `inverse_maps.joblib`
- `feature_cols.joblib`

These files are tracked using Git LFS to handle their large size.

## Environment Variables

The application uses the following environment variables:
- `PORT`: The port number the application should listen on (automatically set by Render)
- `PYTHONUNBUFFERED`: Set to 1 for better logging

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

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd Forecasting-Retail-Consumer-Purchase-Trends
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Server

Run the application using:
```bash
python main.py
```
The server will start on `http://127.0.0.1:8000`

### Using the Web Interface

1. Navigate to `http://127.0.0.1:8000` in your web browser
2. Fill out the form with spending trend information
3. Submit to receive demographic predictions

### Using the API

Make POST requests to `/predict` endpoint with JSON payload:

```json
{
  "Period": "2024-12-20",
  "Location": "MS1",
  "Personal Hygiene (eg Soap, toothpaste)": "Less",
  "Cleaning products": "Less",
  "Long lasting (dry) groceries": "Less",
  "Fresh groceries (Fruits, vegetables)": "Not sure",
  "Medicines/Natural remedies": "Same",
  "Alcohol beverages": "Same",
  "Skin care (eg. Body lotion)": "More",
  "Hair care (eg. Shampoo)": "More",
  "Entertainment (eg. Restaurants, movies)": "More",
  "Electronics (eg Phone, Computers)": "Same",
  "Beauty (eg Makeup, cosmetics, haircuts)": "Same",
  "Clothing": "Same",
  "Airtime/Data bundles": "Not sure"
}
```

### API Response Format

```json
{
  "Gender": "predicted_gender",
  "Age-group": "predicted_age_group",
  "Occupation": "predicted_occupation",
  "Income Level": "predicted_income_level"
}
```

## Input Categories

The system accepts spending trend data for the following categories:
- Personal Hygiene
- Cleaning Products
- Long-lasting Groceries
- Fresh Groceries
- Medicines/Natural Remedies
- Alcohol Beverages
- Skin Care
- Hair Care
- Entertainment
- Electronics
- Beauty Products
- Clothing
- Airtime/Data Bundles

For each category, valid inputs are:
- "More"
- "Less"
- "Same"
- "Not sure"

## Error Handling

The API includes comprehensive error handling for:
- Invalid input data
- Missing required fields
- Model prediction errors
- Server-side processing issues

All errors return appropriate HTTP status codes and detailed error messages.

## Development

### Prerequisites

- Python 3.7+
- FastAPI
- scikit-learn
- pandas
- numpy
- uvicorn

### Running in Development Mode

```bash
uvicorn main:app --reload
```

This enables hot-reloading for development purposes.

## API Documentation

FastAPI provides automatic interactive API documentation:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Authors

[Add author information here]

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

## Streamlit Deployment Instructions

### Handling Large Model Files

This application includes a large model file (`feature_selector.joblib`, 220MB) which exceeds Streamlit's default file size limits. To deploy successfully, follow these steps:

1. **Upload the large model file to Google Drive**:
   - Upload the `feature_selector.joblib` file to Google Drive
   - Make the file publicly accessible by right-clicking and selecting "Share" > "Anyone with the link"
   - Copy the share link

2. **Get a direct download link**:
   - From the share link (which looks like `https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing`)
   - Extract the file ID (the `YOUR_FILE_ID` part)
   - You'll use this ID with the gdown library

3. **Configure Streamlit secrets**:
   - In your Streamlit Cloud dashboard, go to your app settings
   - Add the following to the secrets section:
   ```
   SELECTOR_GDRIVE_URL = "YOUR_FILE_ID"
   ```
   - Note: Enter just the file ID, not the full URL

4. **Set environment variable**:
   - In your Streamlit Cloud dashboard, add an environment variable:
   ```
   STREAMLIT_DEPLOYMENT = cloud
   ```

### Deploying to Streamlit Cloud

1. Push your code to a GitHub repository
2. Log in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Connect to your GitHub repository
5. Select the main file as `app.py`
6. Deploy the app

The application is configured to:
- Load small model files directly from the repository
- Download the large model file (`feature_selector.joblib`) from Google Drive at runtime
- Cache the model to avoid reloading on every interaction

### Troubleshooting

If you encounter issues:
1. Verify the Google Drive file is publicly accessible
2. Check that the file ID in secrets is correct
3. Ensure gdown is properly installed (it's included in requirements.txt)
4. Review Streamlit logs for any download errors
