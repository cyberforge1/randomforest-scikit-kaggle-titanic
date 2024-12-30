# src/data/split.py

import pandas as pd
from sklearn.model_selection import train_test_split
import logging

def split_data():
    # Configure logging for splitting
    logging.basicConfig(
        filename='splitting.log',  # Log to a file named splitting.log
        filemode='a',  # Append mode
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    
    logging.info("Starting data splitting...")
    
    processed_data_path = "data/processed"
    train_processed_path = f"{processed_data_path}/train_processed.csv"
    
    try:
        train_data = pd.read_csv(train_processed_path)
        logging.info("Successfully loaded processed train data.")
    except Exception as e:
        logging.error(f"Error loading processed train data: {e}")
        raise e
    
    # Define features and target
    X = train_data.drop('Survived', axis=1)
    y = train_data['Survived']
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logging.info("Data successfully split into training and validation sets.")
    
    return X_train, X_val, y_train, y_val

if __name__ == "__main__":
    split_data()
