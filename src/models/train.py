# src/models/train.py

import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

def train_model():
    processed_data_path = "data/processed"
    train_processed_path = os.path.join(processed_data_path, "train_processed.csv")
    
    # Check if processed data exists
    if not os.path.exists(train_processed_path):
        raise FileNotFoundError(f"Processed training data not found at {train_processed_path}.")
    
    # Load the processed training data
    train_data = pd.read_csv(train_processed_path)

    # Define features and target
    if 'Survived' not in train_data.columns:
        raise KeyError("'Survived' column not found in the training data.")
    
    X = train_data.drop('Survived', axis=1)
    y = train_data['Survived']

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save the validation set with labels
    validation_data = X_val.copy()
    validation_data['Survived'] = y_val
    validation_save_path = os.path.join(processed_data_path, "validation_set.csv")
    validation_data.to_csv(validation_save_path, index=False)

    # Initialize the Random Forest classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    # Train the model
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = rf_classifier.predict(X_val)

    # Evaluate the model
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)

    # Print evaluation metrics to console
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # Save the trained model to a file
    model_dir = "models"
    model_filename = "random_forest_titanic_model.joblib"
    model_path = os.path.join(model_dir, model_filename)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(rf_classifier, model_path)
    print(f"Trained model saved to {model_path}.")

    # Save the feature order to a file
    feature_order = list(X.columns)
    feature_order_save_path = os.path.join(model_dir, "feature_order.txt")
    with open(feature_order_save_path, 'w') as f:
        for feature in feature_order:
            f.write(f"{feature}\n")

if __name__ == "__main__":
    train_model()
