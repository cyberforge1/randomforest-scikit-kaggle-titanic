# src/models/evaluate_error_rate.py

import pandas as pd
import joblib
import logging
import os
import sys
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

def setup_logging():
    """Set up logging to file and console."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter for log messages
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler
    file_handler = logging.FileHandler('evaluation_error_rate.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def evaluate_error_rate(test_data_path: str):
    """Evaluate the model's error rate on the test dataset."""
    setup_logging()
    logging.info("=== Starting Error Rate Evaluation ===")

    model_path = os.path.join("models", "random_forest_titanic_model.joblib")
    feature_order_path = os.path.join("models", "feature_order.txt")
    
    # Check if model and feature order files exist
    if not os.path.exists(model_path):
        logging.error(f"Trained model not found at {model_path}.")
        raise FileNotFoundError(f"Trained model not found at {model_path}.")
    if not os.path.exists(feature_order_path):
        logging.error(f"Feature order file not found at {feature_order_path}.")
        raise FileNotFoundError(f"Feature order file not found at {feature_order_path}.")
    if not os.path.exists(test_data_path):
        logging.error(f"Test data not found at {test_data_path}.")
        raise FileNotFoundError(f"Test data not found at {test_data_path}.")
    
    try:
        logging.info(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.exception("Failed to load the trained model.")
        raise e
    
    try:
        logging.info(f"Loading feature order from {feature_order_path}...")
        with open(feature_order_path, 'r') as f:
            feature_order = [line.strip() for line in f.readlines()]
        logging.info(f"Feature order: {feature_order}")
    except Exception as e:
        logging.exception("Failed to load feature order.")
        raise e
    
    try:
        logging.info(f"Loading test data from {test_data_path}...")
        test_data = pd.read_csv(test_data_path)
        logging.info("Test data loaded successfully.")
    except Exception as e:
        logging.exception("Failed to load test data.")
        raise e
    
    # Check if 'Survived' column exists
    if 'Survived' not in test_data.columns:
        logging.error("'Survived' column not found in the test data.")
        raise KeyError("'Survived' column not found in the test data.")
    
    try:
        X_test = test_data.drop('Survived', axis=1)
        y_test = test_data['Survived']
        logging.info("Separated features and target variable from test data.")
    except Exception as e:
        logging.exception("Error separating features and target from test data.")
        raise e
    
    try:
        logging.info("Reordering test data features to match training feature order...")
        X_test = X_test[feature_order]
        logging.info("Feature order aligned.")
    except Exception as e:
        logging.exception("Error aligning feature order in test data.")
        raise e
    
    try:
        logging.info("Making predictions on the test set...")
        y_pred = model.predict(X_test)
        logging.info("Predictions completed.")
    except Exception as e:
        logging.exception("Error during prediction on test set.")
        raise e
    
    try:
        logging.info("Calculating evaluation metrics...")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        
        logging.info(f"Validation Accuracy: {accuracy:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1-Score: {f1:.4f}")
        logging.info("Confusion Matrix:")
        logging.info(f"\n{conf_matrix}")
        logging.info("Classification Report:")
        logging.info(f"\n{report}")
        
        # Print metrics to console
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(report)
    except Exception as e:
        logging.exception("Error during evaluation metric calculation.")
        raise e
    
    logging.info("=== Error Rate Evaluation Completed Successfully ===")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Model Error Rate on Test Dataset")
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to the processed test dataset CSV file (e.g., data/processed/test_processed.csv)"
    )
    args = parser.parse_args()

    try:
        evaluate_error_rate(args.test_data)
    except Exception as e:
        logging.exception("Evaluation failed.")
        sys.exit(1)
