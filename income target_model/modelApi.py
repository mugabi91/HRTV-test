from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from scipy.sparse import hstack

app = FastAPI(title="Risk Model Prediction API")

# Define the input data model
class PredictionRequest(BaseModel):
    HH_Income: float  # Represents "HH Income + Production/Day (USD)"
    most_recommend_rtv_program: str
    least_recommend_rtv_program: str
    most_recommend_rtv_program_reason: str
    least_recommend_rtv_program_reason: str

# Global variables to hold the loaded objects
model = None
preprocessor = None
vectorizer = None

@app.on_event("startup")
def load_models():
    """
    Load the trained model, preprocessor, and vectorizer on startup.
    """
    global model, preprocessor, vectorizer
    try:
        model = joblib.load("best_model.pkl")
        preprocessor = joblib.load("preprocessor.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
    except Exception as e:
        raise RuntimeError(f"Failed to load model or transformers: {e}")

def load_and_predict_api(data: PredictionRequest) -> str:
    """
    Create a DataFrame from the incoming data, transform it using the loaded
    preprocessor and vectorizer, and use the model to generate a prediction.
    Returns a human-readable prediction.
    """
    # Build a DataFrame from the input
    df_input = pd.DataFrame({
        "HH Income + Production/Day (USD)": [data.HH_Income],
        "most_recommend_rtv_program": [data.most_recommend_rtv_program],
        "least_recommend_rtv_program": [data.least_recommend_rtv_program],
        "most_recommend_rtv_program_reason": [data.most_recommend_rtv_program_reason],
        "least_recommend_rtv_program_reason": [data.least_recommend_rtv_program_reason]
    })
    
    # Separate numerical/categorical features and text features
    X_new = df_input[["HH Income + Production/Day (USD)", 
                      "most_recommend_rtv_program", 
                      "least_recommend_rtv_program"]]
    text_column = df_input[["most_recommend_rtv_program_reason", "least_recommend_rtv_program_reason"]]
    
    # Transform numerical & categorical features
    X_transformed = preprocessor.transform(X_new)
    
    # Combine text columns into a single string per row
    text_series = text_column.fillna("Unknown").astype(str).apply(lambda row: " ".join(row), axis=1)
    X_text_transformed = vectorizer.transform(text_series)
    
    # Combine the transformed features
    X_final = hstack([X_transformed, X_text_transformed])
    
    # To maintain consistency, convert the final features to a DataFrame with feature names
    pre_cols = preprocessor.get_feature_names_out()
    vec_cols = vectorizer.get_feature_names_out()
    combined_cols = np.concatenate([pre_cols, vec_cols])
    X_final_df = pd.DataFrame(X_final.todense(), columns=combined_cols)
    
    # Get prediction from the model
    predictions = model.predict(X_final_df)
    mapping = {0: "Not at risk", 1: "At risk"}
    return mapping.get(predictions[0], "Unknown")

@app.post("/predict")
def predict(request: PredictionRequest):
    """
    API endpoint to get risk predictions.
    """
    try:
        result = load_and_predict_api(request)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

if __name__ == "__main__":
    import uvicorn
    # Run the API on host 0.0.0.0 and port 8000
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
