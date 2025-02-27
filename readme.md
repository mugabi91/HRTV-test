# Risk Model Prediction API

This project implements a production-ready API to predict whether a household is at risk of not achieving the \$2/day income target. The API uses a pre-trained machine learning model along with data preprocessing and text vectorization steps. It is built using FastAPI and is designed to be easily integrated into larger systems.

## Overview

- **Objective:** Identify households at risk by predicting if their daily income is below \$2.
- **Data Sources:** The model uses a combination of numerical, categorical, and textual features:
  - **Numerical:** `HH Income + Production/Day (USD)`
  - **Categorical:** `most_recommend_rtv_program`, `least_recommend_rtv_program`
  - **Text:** `most_recommend_rtv_program_reason`, `least_recommend_rtv_program_reason`

- **Features:**
  - Data preprocessing (imputation, scaling, one-hot encoding)
  - Text feature extraction using TF-IDF
  - Model training with hyperparameter tuning using GridSearchCV
  - An API built with FastAPI that accepts JSON input and returns a prediction ("At risk" or "Not at risk").

## Features

- **Robust Data Preprocessing:** Handles missing values, categorical encoding, and numerical scaling.
- **Text Embedding:** Converts textual feedback into numerical features with TF-IDF.
- **Hyperparameter Tuning:** Selects the best model using cross-validation.
- **Model Retraining:** Option to automatically retrain the model if new data becomes available.
- **FastAPI Integration:** Provides a RESTful API for easy integration with frontend systems or other services.

## Requirements

- Python 3.7+
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Joblib](https://joblib.readthedocs.io/)
- [Rich](https://rich.readthedocs.io/)
- [SciPy](https://scipy.org/)

Install the dependencies using pip:

```bash
pip install fastapi uvicorn scikit-learn pandas numpy joblib rich scipy
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/mugabi91/HRTV-test
   cd HRTV-test
   ```

2. Ensure you have all dependencies installed (see **Requirements**).

3. Place your training dataset (e.g., `interview_dataset.csv`) in the designated folder or adjust the file paths in the script.

## Running the API

Start the API using Uvicorn:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoint

### POST `/predict`

- **Description:** Returns a risk prediction based on the provided household data.
- **Input Format (JSON):**

  ```json
  {
    "HH_Income": 1.5,
    "most_recommend_rtv_program": "Program A",
    "least_recommend_rtv_program": "Program X",
    "most_recommend_rtv_program_reason": "Great content and community support",
    "least_recommend_rtv_program_reason": "Boring and uninformative"
  }
  ```

- **Output Format (JSON):**

  ```json
  {
    "prediction": "At risk"
  }
  ```

## How It Works

1. **Model & Transformers Loading:**  
   On startup, the API loads the pre-trained model, preprocessor, and TF-IDF vectorizer from disk.

2. **Data Transformation:**  
   Incoming JSON data is converted to a DataFrame. Numerical and categorical features are transformed using the preprocessor, while the text fields are combined and vectorized using TF-IDF.

3. **Prediction:**  
   The transformed features are concatenated and passed to the model to generate a prediction. The numeric prediction (0 or 1) is then mapped to "Not at risk" or "At risk".

4. **Retraining:**  
   The system supports automatic retraining if a new data file (`new_data.csv`) is detected. This ensures the model remains up-to-date as new data becomes available.

## MLOps Considerations

- **Data Pipeline:**  
  New feedback can be integrated by placing a file in the designated folder or connecting to a data warehouse, triggering a retraining cycle.
  
- **Monitoring & Logging:**  
  The API uses Python’s logging module to track key events and errors for easier debugging and monitoring in production environments.

- **Scalability:**  
  The modular design makes it easy to extend functionality, such as adding more endpoints, integrating with external services, or deploying to cloud platforms.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests for any enhancements or bug fixes
