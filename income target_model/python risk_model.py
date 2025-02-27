"""
    Production Ready Script for Risk Model Training & Retraining

    This script:
    - Loads and preprocesses the data.
    - Splits the data into training and test sets.
    - Performs hyperparameter tuning with GridSearchCV on several models.
    - Selects and evaluates the best model on the test set.
    - Saves the best model, preprocessor, and vectorizer.
    - Optionally retrains the model if new data is available.

Usage: 
    python risk_model.py
"""

import os
import json
import logging
from rich.logging import RichHandler
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Configure logging
logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

log = logging.getLogger("rich")

# Global file paths (adjust these paths as needed in production)
DATA_FILE = "interview_dataset.csv"  # Main dataset
NEW_DATA_FILE = "income target_model/new_data.csv"  # New data for retraining
MODEL_FILE = "income target_model/best_model.pkl"   # Trained model file
PREPROCESSOR_FILE = "income target_model/preprocessor.pkl"  # Preprocessor file for data transformation
VECTORIZER_FILE = "income target_model/vectorizer.pkl"  # Text vectorizer file

def load_data(file_path):
    """
    Load CSV data from the given file path using selected columns.
    Returns a pandas DataFrame or None if an error occurs.
    """
    try:
        if not os.path.exists(file_path):
            log.error(f"File {file_path} not found.")
            return None
        df = pd.read_csv(
            file_path,
            usecols=[
                "HH Income + Production/Day (USD)",  # numeric
                "most_recommend_rtv_program",         # categorical
                "least_recommend_rtv_program",        # categorical
                "most_recommend_rtv_program_reason",  # text
                "least_recommend_rtv_program_reason"  # text
            ]
        )
        log.info(f"Data loaded successfully from {file_path}.")
        return df
    except pd.errors.EmptyDataError:
        logging.error(f"File {file_path} is empty.")
    except pd.errors.ParserError:
        log.error(f"Error parsing file {file_path}.")
    except Exception as e:
        log.error(f"An error occurred while loading data: {e}")
    return None

def preprocess_data(df, training=True):
    """
    Preprocess the data by imputing missing values, encoding categorical variables,
    scaling numerical features, and extracting text features using TF-IDF.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
        training (bool): Whether to fit transformers (True for training, False for test data).
    
    Returns:
        X_final (pd.DataFrame): Preprocessed feature dataframe.
        y (pd.Series): Target variable.
        preprocessor (ColumnTransformer): Fitted preprocessor for numerical/categorical data.
        vectorizer (TfidfVectorizer): Fitted text vectorizer.
    """
    # Define target: 1 if income is less than $2/day, else 0
    df['risk_target'] = (df['HH Income + Production/Day (USD)'] < 2).astype(int)
    X = df.drop(columns=['risk_target'])
    y = df['risk_target']
    
    # Define columns
    num_cols = ['HH Income + Production/Day (USD)']
    cat_cols = ['most_recommend_rtv_program', 'least_recommend_rtv_program']
    text_cols = ['most_recommend_rtv_program_reason', 'least_recommend_rtv_program_reason']
    
    # Set up imputers, encoder, and scaler
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler()
    
    # Set up text feature extraction
    vectorizer = TfidfVectorizer(max_features=100)
    
    # Pipelines for numerical and categorical data
    num_pipeline = Pipeline([("imputer", num_imputer), ("scaler", scaler)])
    cat_pipeline = Pipeline([("imputer", cat_imputer), ("encoder", onehot_encoder)])
    
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])
    
    # Fit or transform the data
    X_processed = preprocessor.fit_transform(X) if training else preprocessor.transform(X)
    
    # Process text features by concatenating text columns
    text_series = X[text_cols].fillna("Unknown").astype(str).apply(lambda row: " ".join(row), axis=1)
    text_features = vectorizer.fit_transform(text_series).toarray() if training else vectorizer.transform(text_series).toarray()
    text_feature_names = vectorizer.get_feature_names_out()
    text_df = pd.DataFrame(text_features, columns=text_feature_names)
    
    # Convert preprocessed array to DataFrame with column names
    transformed_columns = preprocessor.get_feature_names_out()
    X_processed_df = pd.DataFrame(X_processed, columns=transformed_columns)
    
    # Combine numerical/categorical features with text features
    X_final = pd.concat([X_processed_df, text_df], axis=1)
    
    return X_final, y, preprocessor, vectorizer

def train_and_select_best_model(X_train, y_train, X_test, y_test):
    """
    Train multiple models using GridSearchCV and select the one with the best test accuracy.
    
    Returns:
        best_model (sklearn estimator): The best performing model.
    """
    param_grids = {
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }
        },
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=500),
            "params": {
                "C": [0.1, 1, 10],
                "solver": ["liblinear", "lbfgs"]
            }
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 10]
            }
        },
        "Support Vector Machine": {
            "model": SVC(),
            "params": {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"]
            }
        },
        "K-Nearest Neighbors": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [3, 5, 7],
                "weights": ["uniform", "distance"]
            }
        },
        "Neural Network": {
            "model": MLPClassifier(max_iter=500),
            "params": {
                "hidden_layer_sizes": [(50,), (100,), (50, 50)],
                "activation": ["relu", "tanh"],
                "alpha": [0.0001, 0.001]
            }
        }
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    best_params = {}
    
    for name, config in param_grids.items():
        model = config["model"]
        params = config["params"]
        grid_search = GridSearchCV(model, params, cv=5, scoring="accuracy", n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        model_best_params = grid_search.best_params_
        best_model_cv = grid_search.best_estimator_
        best_score_cv = grid_search.best_score_
        
        # Evaluate on test set
        test_score = best_model_cv.score(X_test, y_test)
        logging.info(f"{name}: CV Accuracy = {best_score_cv:.4f} | Test Accuracy = {test_score:.4f} | Params = {model_best_params}")
        
        y_pred = best_model_cv.predict(X_test)
        logging.info(f"\n📋 {name} Classification Report:\n{classification_report(y_test, y_pred)}")
        
        if test_score > best_score:
            best_score = test_score
            best_model = best_model_cv
            best_name = name
            best_params = model_best_params
    
    # Save best model parameters to JSON file
    with open("income target_model/best_model_params.json", "w") as f:
        json.dump({"model": best_name, "accuracy": best_score, "params": best_params}, f, indent=4)
    
    log.info(f"✅ Best Model: {best_name} with Test Accuracy = {best_score:.4f}")
    log.info("📄 Best Parameters saved in 'best_model_params.json'")
    return best_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the given model on the test set and print out evaluation metrics.
    """
    y_pred = model.predict(X_test)
    log.info("📊 Model Evaluation on Test Set:")
    log.info(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f} | "
        f"🎯 Precision: {precision_score(y_test, y_pred, average='weighted'):.4f} | "
        f"🔄 Recall: {recall_score(y_test, y_pred, average='weighted'):.4f} | "
        f"🏆 F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}"
        )
    log.info("\n📋 Classification Report:")
    log.info(classification_report(y_test, y_pred))

def retrain_if_new_data():
    """
    Check if new data exists for retraining; if so, retrain the current model.
    """
    if os.path.exists(NEW_DATA_FILE):
        log.info("New data found. Retraining model...")
        new_df = load_data(NEW_DATA_FILE)
        if new_df is None:
            log.error("New Data file is invalid. Check data file.")
            return
        X_new, y_new, _, _ = preprocess_data(new_df, training=False)
        model = joblib.load(MODEL_FILE)
        model.fit(X_new, y_new)
        joblib.dump(model, MODEL_FILE)
        log.info("Model retrained and updated.")
        os.remove(NEW_DATA_FILE)
    else:
        log.info("No new data found. Model remains unchanged.")

def load_and_predict(X_new, text_column):
    """
    Load preprocessor, vectorizer, and trained model to generate predictions on new data.
    
    Parameters:
        X_new (pd.DataFrame): DataFrame containing numerical & categorical features.
        text_column (pd.DataFrame): DataFrame containing text features.
        
    Returns:
        predictions: Model predictions.
    """
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    X_transformed = preprocessor.transform(X_new)
    
    vectorizer = joblib.load(VECTORIZER_FILE)
    # Combine text columns as during training
    text_series = text_column.fillna("Unknown").astype(str).apply(lambda row: " ".join(row), axis=1)
    X_text_transformed = vectorizer.transform(text_series)
    
    X_final = hstack([X_transformed, X_text_transformed])
    pre_cols = preprocessor.get_feature_names_out()
    vec_cols = vectorizer.get_feature_names_out()
    combined_cols = np.concatenate([pre_cols, vec_cols])
    X_final_df = pd.DataFrame(X_final.todense(), columns=combined_cols)
    
    model = joblib.load(MODEL_FILE)
    predictions = model.predict(X_final_df)
    return predictions

def map_predictions(predictions):
    """
    Map numerical predictions (0, 1) to human-readable labels.
    """
    mapping = {0: "Not at risk", 1: "At risk"}
    mapped_predictions = [mapping.get(pred, "Unknown") for pred in predictions]
    log.info("Mapped Predictions: %s", mapped_predictions)
    return mapped_predictions

def main():
    # Load main dataset
    df = load_data(DATA_FILE)
    if df is None:
        log.error("Data loading failed. Exiting.")
        exit(1)
    
    # Preprocess data
    X, y, preprocessor, vectorizer = preprocess_data(df)
    
    # Split data (80% training, 20% testing) with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )
    
    # Train and select the best model
    best_model = train_and_select_best_model(X_train, y_train, X_test, y_test)
    
    # Evaluate model on test set
    evaluate_model(best_model, X_test, y_test)
    
    # Save model, preprocessor, and vectorizer
    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(preprocessor, PREPROCESSOR_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    log.info("Model training complete. Best model and transformers saved.")
    
    # Retrain model if new data is available
    retrain_if_new_data()

if __name__ == "__main__":
    main()
