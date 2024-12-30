# main.py

import argparse
from src.models.evaluate import evaluate_model
from src.utils.helpers import parse_input
import logging
import sys

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
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def main():
    # Initialize logging
    setup_logging()
    logging.info("=== Starting Main Evaluation Script ===")
    
    parser = argparse.ArgumentParser(description="Interact with the Titanic Survival Model")
    parser.add_argument("--input", type=str, required=True, help="Input features in the format 'age=22,sex=female,class=3,SibSp=0,Parch=0,Fare=7.8292,Embarked=S'")
    args = parser.parse_args()

    try:
        input_features = parse_input(args.input)
        logging.info(f"Parsed Input Features: {input_features}")
    except Exception as e:
        logging.exception("Failed to parse input features.")
        sys.exit(1)

    try:
        prediction = evaluate_model(input_features)
        result = 'Survived' if prediction == 1 else 'Did Not Survive'
        logging.info(f"Survival Prediction: {result}")
        print(f"Survival Prediction: {result}")
    except Exception as e:
        logging.exception("Failed to evaluate the model.")
        sys.exit(1)
    
    logging.info("=== Evaluation Completed Successfully ===")

if __name__ == "__main__":
    main()
