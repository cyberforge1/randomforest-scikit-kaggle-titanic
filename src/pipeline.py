# src/pipeline.py

from src.data.download import download_titanic_dataset
from src.data.preprocess import preprocess_data
from src.models.train import train_model

def run_pipeline():
    download_titanic_dataset()
    preprocess_data()
    train_model()

if __name__ == "__main__":
    run_pipeline()
