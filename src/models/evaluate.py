# src/models/evaluate.py

import pandas as pd
import joblib
import logging
import os

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
    file_handler = logging.FileHandler('evaluation.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def evaluate_model(input_features: dict):
    """Load the trained model and make a prediction based on input features."""
    setup_logging()
    logging.info("=== Starting Model Evaluation ===")
    
    model_path = os.path.join("models", "random_forest_titanic_model.joblib")
    feature_order_path = os.path.join("models", "feature_order.txt")
    
    # Check if both model and feature order files exist
    if not os.path.exists(model_path):
        logging.error(f"Trained model not found at {model_path}.")
        raise FileNotFoundError(f"Trained model not found at {model_path}.")
    if not os.path.exists(feature_order_path):
        logging.error(f"Feature order file not found at {feature_order_path}.")
        raise FileNotFoundError(f"Feature order file not found at {feature_order_path}.")
    
    try:
        logging.info(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.exception("Failed to load the trained model.")
        raise e
    
    try:
        logging.info("Loading feature order...")
        with open(feature_order_path, 'r') as f:
            feature_order = [line.strip() for line in f.readlines()]
        logging.info(f"Feature order: {feature_order}")
    except Exception as e:
        logging.exception("Failed to load feature order.")
        raise e
    
    try:
        logging.info("Preparing input data for prediction...")
        input_df = pd.DataFrame([input_features])
        logging.info(f"Initial Input DataFrame:\n{input_df}")
        
        # Reorder columns to match feature order
        input_df = input_df[feature_order]
        logging.info(f"Reordered Input DataFrame:\n{input_df}")
    except Exception as e:
        logging.exception("Error preparing input data.")
        raise e
    
    try:
        logging.info("Making prediction...")
        prediction = model.predict(input_df)
        logging.info(f"Prediction result: {prediction[0]}")
    except Exception as e:
        logging.exception("Error during prediction.")
        raise e
    
    logging.info("=== Model Evaluation Completed ===")
    return prediction[0]
