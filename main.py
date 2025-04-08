import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from typing import Dict, Any, List
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request

# --- Configuration ---
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, 'multi_output_model.joblib')
ENCODERS_PATH = os.path.join(MODEL_DIR, 'label_encoders.joblib')
SELECTOR_PATH = os.path.join(MODEL_DIR, 'feature_selector.joblib')
# SELECTED_FEATURES_PATH = os.path.join(MODEL_DIR, 'selected_features.joblib') # Not strictly needed if using selector object
INVERSE_MAPS_PATH = os.path.join(MODEL_DIR, 'inverse_maps.joblib')
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, 'feature_cols.joblib')

# --- Load Artifacts on Startup ---
try:
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Model directory '{MODEL_DIR}' not found. Please ensure it exists and contains the saved artifacts.")

    print(f"Loading model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"Loading label encoders from: {ENCODERS_PATH}")
    label_encoders = joblib.load(ENCODERS_PATH)
    print(f"Loading feature selector from: {SELECTOR_PATH}")
    selector = joblib.load(SELECTOR_PATH)
    print(f"Loading inverse maps from: {INVERSE_MAPS_PATH}")
    inverse_maps = joblib.load(INVERSE_MAPS_PATH)
    print(f"Loading original feature columns from: {FEATURE_COLS_PATH}")
    original_feature_cols = joblib.load(FEATURE_COLS_PATH) # Get the expected input columns

    print("--- Model artifacts loaded successfully ---")
    print(f"Expecting features (order matters for DataFrame creation): {original_feature_cols}")
    print(f"Label Encoders loaded for: {list(label_encoders.keys())}")
   

    print(f"Inverse Maps loaded for targets: {list(inverse_maps.keys())}")
    print("--- End loading ---")

except FileNotFoundError as e:
    print(f"Error loading model artifacts: {e}")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred during artifact loading: {e}")
    # Optional: import traceback; traceback.print_exc()
    exit(1)


# --- Define Input Data Model using Pydantic ---
# Dynamically creating this might be more robust if features change often,
# but explicitly listing them is clearer for this example.
class InputFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True)  # V2 syntax

    Period: str = Field(..., example="2024-12-20")
    Location: str = Field(..., example="MS1")
    Personal_Hygiene__eg_Soap__toothpaste_: str = Field(..., 
        alias="Personal Hygiene (eg Soap, toothpaste)", 
        example="Less")
    Cleaning_products: str = Field(..., alias="Cleaning products", example="Less")
    Long_lasting__dry__groceries: str = Field(..., alias="Long lasting (dry) groceries", example="Less")
    Fresh_groceries__Fruits__vegetables_: str = Field(..., alias="Fresh groceries (Fruits, vegetables)", example="Not sure")
    Medicines_Natural_remedies: str = Field(..., alias="Medicines/Natural remedies", example="Same")
    Alcohol_beverages: str = Field(..., alias="Alcohol beverages", example="Same")
    Skin_care__eg__Body_lotion_: str = Field(..., alias="Skin care (eg. Body lotion)", example="More")
    Hair_care__eg__Shampoo_: str = Field(..., alias="Hair care (eg. Shampoo)", example="More")
    Entertainment__eg__Restaurants__movies_: str = Field(..., alias="Entertainment (eg. Restaurants, movies)", example="More")
    Electronics__eg_Phone__Computers_: str = Field(..., alias="Electronics (eg Phone, Computers)", example="Same")
    Beauty__eg_Makeup__cosmetics__haircuts_: str = Field(..., alias="Beauty (eg Makeup, cosmetics, haircuts)", example="Same")
    Clothing: str = Field(..., example="Same")
    Airtime_Data_bundles: str = Field(..., alias="Airtime/Data bundles", example="Not sure")


# --- Define Output Data Model ---
class PredictionOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)  # V2 syntax

    Gender: str
    Age_group: str = Field(..., alias="Age-group", description="Age group category")
    Occupation: str
    Income_Level: str = Field(..., alias="Income Level", description="Income level category")


# --- Create FastAPI App ---
app = FastAPI(
    title="Demographic Prediction API",
    description="Predicts Gender, Age Group, Occupation, and Income Level based on spending habits.",
    version="1.0.1" # Incremented version
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Define form fields and options
FORM_FIELDS = [
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

DROPDOWN_OPTIONS = {
    field["name"]: ["More", "Less", "Same", "Not sure"] for field in FORM_FIELDS
}
# Special options for Period and Location
DROPDOWN_OPTIONS["Period"] = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"]
DROPDOWN_OPTIONS["Location"] = ["MS1", "MS2", "MS3", "MS4", "MS5"]

# Add a route to serve the HTML page
@app.get("/", tags=["UI"])
async def serve_page(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "form_fields": FORM_FIELDS,
            "dropdown_options": DROPDOWN_OPTIONS
        }
    )

# --- Health Check Endpoint ---
@app.get("/", tags=["Health Check"])
async def read_root():
    return {"status": "API is running!"}


# --- Prediction Endpoint ---
@app.post("/predict", response_model=PredictionOut, tags=["Prediction"])
async def predict(features: InputFeatures):
    """
    Performs inference based on input features.

    Takes a JSON object with spending habit features and returns predicted
    demographics.
    """
    print("\n--- Received Prediction Request ---")
    try:
        # 1. Convert Pydantic model to DataFrame, using original names
        # Use .dict(by_alias=True) to get keys matching original column names
        input_dict = features.dict(by_alias=True)
        print(f"Input Data (dict): {input_dict}")
        input_data = pd.DataFrame([input_dict])

        # Ensure the DataFrame has columns in the same order as original_feature_cols
        # This is crucial for the model if column order matters during training/prediction
        try:
            input_data = input_data[original_feature_cols]
            print(f"DataFrame columns ordered: {input_data.columns.tolist()}")
        except KeyError as e:
            raise ValueError(f"Input data is missing expected feature column: {e}")

        # 2. Preprocessing - Apply Label Encoding
        input_encoded = input_data.copy()
        print("Applying Label Encoders...")
        for col, le in label_encoders.items():
            if col in input_encoded.columns:
                print(f"  Encoding column: '{col}'")
                # Handle potential unseen values during transform
                current_col_data = input_encoded[col].astype(str) # Ensure string type
                known_classes = set(le.classes_)
                unknown_marker = -1 # Define how to handle unknowns numerically
                encoded_values = []

                for item in current_col_data:
                     if item in known_classes:
                         encoded_values.append(le.transform([item])[0])
                     else:
                         print(f"  Warning: Unseen value '{item}' in column '{col}'. Mapping to {unknown_marker}.")
                         # Option 1: Assign a default like -1 (check if model handles negatives)
                         encoded_values.append(unknown_marker)
                         

                input_encoded[col] = encoded_values
                # print(f"    Encoded values for '{col}': {input_encoded[col].tolist()}") # Verbose debug
            else:
                 # This shouldn't happen if input_data has original_feature_cols and encoders match
                 print(f"Warning: Column '{col}' from label encoder not found in input data DataFrame.")

        print("Label Encoding Complete.")
        print(f"Encoded DataFrame head:\n{input_encoded.head().to_string()}")

        # 3. Feature Selection (using the loaded selector object)
        print("Applying Feature Selector...")
        try:
            # The selector expects input with the same columns (and order) as it was fit on.
            # Ensure input_encoded matches that structure.
            input_selected = selector.transform(input_encoded)

            # Verify selected features (optional, for debugging)
            try:
                 selected_feature_names_from_selector = input_encoded.columns[selector.get_support()]
                 print(f"Applied feature selector. Selected features: {selected_feature_names_from_selector.tolist()}")
            except Exception as e:
                 print(f"Could not get selected feature names directly from selector: {e}") # Informational

            print(f"Shape after selection: {input_selected.shape}")
        except ValueError as e:
             # This might happen if columns mismatch or if unseen encoded values (-1) cause issues
             print(f"Error during feature selection transform: {e}")
             print(f"Input data shape to selector: {input_encoded.shape}")
             print(f"Input data columns to selector: {input_encoded.columns.tolist()}")
             # Potentially check selector.n_features_in_ vs input_encoded.shape[1]
             raise HTTPException(status_code=400, detail=f"Error applying feature selection, possibly due to data mismatch or unseen encoded value: {e}")
        except Exception as e:
             # Catch other unexpected errors during selection
             print(f"Unexpected error during feature selection: {e}")
             raise HTTPException(status_code=500, detail=f"Internal Server Error during feature selection: {e}")

        # 4. Make Prediction
        print("Making prediction...")
        prediction_encoded = model.predict(input_selected) # Returns shape like (1, N_targets)
        print(f"Raw prediction (encoded): {prediction_encoded}")

        # Check if prediction is valid
        if prediction_encoded is None or len(prediction_encoded) == 0 or len(prediction_encoded[0]) == 0:
             print("Error: Model prediction failed or returned empty.")
             raise HTTPException(status_code=500, detail="Model prediction failed or returned empty.")

        prediction_values = prediction_encoded[0] # Get the first (and only) row of predictions

        # 5. Inverse Transform Predictions
        print("Inverse transforming predictions...")
        results = {}

        # IMPORTANT: Define the expected order of target outputs from your model.
        # This MUST match the order of columns in y_train when the model was trained.
        # Based on your training script: y = encoded_data[['Gender', 'Age-group', 'Occupation', "Income Level"]]
        expected_target_order = ['Gender', 'Age-group', 'Occupation', 'Income Level']
        print(f"Expected target order for decoding: {expected_target_order}")

        # Verify loaded inverse_maps keys match (or can be mapped to) the expected order
        if set(inverse_maps.keys()) != set(expected_target_order):
            print(f"Warning: Inverse map keys {list(inverse_maps.keys())} do not perfectly match expected target order {expected_target_order}. Ensure mapping is correct.")
            # If keys differ significantly, manual mapping might be needed. Assuming keys match for now.


        if len(prediction_values) != len(expected_target_order):
            print(f"Error: Mismatch between number of predicted values ({len(prediction_values)}) and expected targets ({len(expected_target_order)}).")
            raise HTTPException(status_code=500,
                                detail=f"Prediction Error: Model output size ({len(prediction_values)}) doesn't match expected number of targets ({len(expected_target_order)}). Verify model structure.")

        for i, target_name in enumerate(expected_target_order):
            encoded_val = prediction_values[i]
            inv_map = inverse_maps.get(target_name)

            if inv_map is None:
                 print(f"Error: No inverse map found for target '{target_name}'. Cannot decode.")
                 decoded_val = f"Error: Missing map for {target_name}" # Placeholder error value
            else:
                # Use .get(key, default) for safer lookup within the specific inverse map
                decoded_val = inv_map.get(encoded_val, f"Unknown code ({encoded_val})") # Provide default if key missing
            print(f"  Decoding: Target='{target_name}', Encoded={encoded_val}, Decoded='{decoded_val}'")
            results[target_name] = decoded_val # Use original target name as key ('Age-group', 'Income Level')

        print(f"Final results dictionary: {results}")

        # 6. Return results using the Pydantic output model
        print("Constructing output model...")
        # Use dictionary unpacking (**results). Pydantic will map keys (matching aliases)
        # to the model fields because allow_population_by_field_name = True.
        try:
            # Simpler validation - just check if all expected target names are present
            if not all(key in results for key in expected_target_order):
                missing_keys = set(expected_target_order) - set(results.keys())
                print(f"Internal Error: Missing expected keys in results dict before creating output model: {missing_keys}")
                raise ValueError(f"Internal Error: Missing expected keys in results dict: {missing_keys}")

            output_model = PredictionOut(**results)  # Unpack dict, Pydantic maps keys using aliases
            print(f"Output model created successfully: {output_model.model_dump(by_alias=True)}")  # Changed from dict() to model_dump()
            print("--- Prediction Request Complete ---")
            return output_model

        except ValidationError as pydantic_error:
             # Catch Pydantic validation errors during creation more specifically
             print(f"Error creating PredictionOut model: {pydantic_error}")
             print(f"Results dictionary passed: {results}") # Log the data passed
             error_details = pydantic_error.errors()
             raise HTTPException(status_code=500, detail={"message": "Internal Server Error: Could not construct output model.", "data_passed": results, "validation_errors": error_details})
        except Exception as e:
             # Catch other errors during output model creation
             print(f"Unexpected error creating PredictionOut model: {e}")
             print(f"Results dictionary passed: {results}")
             raise HTTPException(status_code=500, detail=f"Internal Server Error: Could not construct output model. Error: {e}")


    # --- Outer Exception Handlers ---
    except ValueError as e:
        # Specific handling for errors raised explicitly (e.g., unseen values, missing columns)
        print(f"Input Data/Value Error: {e}")
        raise HTTPException(status_code=400, detail=f"Input Data Error: {e}")
    except KeyError as e:
        # Handle missing keys if not caught earlier (e.g., unexpected dict key access)
        print(f"Key Error during processing: {e}")
        raise HTTPException(status_code=400, detail=f"Data Processing Error: Missing expected key: {e}")
    except HTTPException as e:
        # Re-raise HTTPExceptions raised within the try block
        raise e
    except Exception as e:
        # Catch-all for other unexpected errors during prediction
        print(f"--- Unexpected Prediction Error ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        # Optional: Log the full traceback for server-side debugging
        import traceback
        traceback.print_exc()
        print(f"--- End Error ---")
        raise HTTPException(status_code=500, detail=f"Internal Server Error during prediction processing: {type(e).__name__}")


# --- Run the API using Uvicorn (for development) ---
if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server...")
    # Use reload=True for development to auto-reload on code changes
    # Use host="0.0.0.0" to make it accessible on your network (use with caution)
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)