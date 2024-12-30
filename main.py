# main.py

import argparse
from src.models.evaluate import evaluate_model
from src.utils.helpers import parse_input

def main():
    parser = argparse.ArgumentParser(description="Interact with the Titanic Survival Model")
    parser.add_argument("--input", type=str, required=True, help="Input features in the format 'age=22,sex=female,class=3'")
    args = parser.parse_args()

    input_features = parse_input(args.input)
    prediction = evaluate_model(input_features)

    print(f"Survival Prediction: {'Survived' if prediction == 1 else 'Did Not Survive'}")

if __name__ == "__main__":
    main()
