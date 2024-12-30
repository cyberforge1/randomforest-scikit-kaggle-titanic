# src/models/train.py

import pandas as pd
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

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
    file_handler = logging.FileHandler('training.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def train_model():
    # Initialize logging
    setup_logging()
    logging.info("=== Starting Model Training ===")

    processed_data_path = "data/processed"
    train_processed_path = os.path.join(processed_data_path, "train_processed.csv")
    
    # Check if processed data exists
    if not os.path.exists(train_processed_path):
        logging.error(f"Processed training data not found at {train_processed_path}.")
        raise FileNotFoundError(f"Processed training data not found at {train_processed_path}.")
    
    try:
        # Load the processed training data
        logging.info(f"Loading processed training data from {train_processed_path}...")
        train_data = pd.read_csv(train_processed_path)
        logging.info("Successfully loaded processed training data.")
    except Exception as e:
        logging.exception("Failed to load processed training data.")
        raise e

    # Define features and target
    if 'Survived' not in train_data.columns:
        logging.error("'Survived' column not found in the training data.")
        raise KeyError("'Survived' column not found in the training data.")
    
    X = train_data.drop('Survived', axis=1)
    y = train_data['Survived']
    logging.info("Separated features and target variable.")

    # Split the data into training and validation sets
    try:
        logging.info("Splitting data into training and validation sets (80-20 split)...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logging.info(f"Data split complete. Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

        # Save the validation set with labels
        validation_data = X_val.copy()
        validation_data['Survived'] = y_val
        validation_save_path = os.path.join(processed_data_path, "validation_set.csv")
        validation_data.to_csv(validation_save_path, index=False)
        logging.info(f"Validation set saved to {validation_save_path}.")
    except Exception as e:
        logging.exception("Failed to split data into training and validation sets.")
        raise e

    # Initialize the Random Forest classifier
    logging.info("Initializing Random Forest Classifier...")
    rf_classifier = RandomForestClassifier(
        n_estimators=100,  # Number of trees in the forest
        random_state=42,    # Seed for reproducibility
        n_jobs=-1,          # Use all available cores
        verbose=1           # Enable verbose output
    )
    logging.info("Random Forest Classifier initialized.")

    try:
        # Train the model
        logging.info("Starting model training...")
        rf_classifier.fit(X_train, y_train)
        logging.info("Random Forest model training completed.")
    except Exception as e:
        logging.exception("Error during model training.")
        raise e

    try:
        # Make predictions on the validation set
        logging.info("Making predictions on the validation set...")
        y_pred = rf_classifier.predict(X_val)
        logging.info("Predictions completed.")

        # Evaluate the model
        logging.info("Evaluating model performance...")
        accuracy = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred)
        logging.info(f"Validation Accuracy: {accuracy:.4f}")
        logging.info("Classification Report:")
        logging.info(f"\n{report}")

        # Print evaluation metrics to console
        print(f"Validation Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
    except Exception as e:
        logging.exception("Error during model evaluation.")
        raise e

    # Save the trained model to a file
    model_dir = "models"
    model_filename = "random_forest_titanic_model.joblib"
    model_path = os.path.join(model_dir, model_filename)
    os.makedirs(model_dir, exist_ok=True)

    try:
        logging.info(f"Saving the trained model to {model_path}...")
        joblib.dump(rf_classifier, model_path)
        logging.info(f"Trained model saved to {model_path}.")
        print(f"Trained model saved to {model_path}.")
    except Exception as e:
        logging.exception("Error saving the trained model.")
        raise e

    # Save the feature order to a file
    try:
        feature_order = list(X.columns)
        feature_order_save_path = os.path.join(model_dir, "feature_order.txt")
        with open(feature_order_save_path, 'w') as f:
            for feature in feature_order:
                f.write(f"{feature}\n")
        logging.info(f"Feature order saved to {feature_order_save_path}.")
    except Exception as e:
        logging.exception("Error saving the feature order.")
        raise e

    logging.info("=== Model Training Pipeline Completed Successfully ===")

if __name__ == "__main__":
    train_model()
