# src/data/download.py

import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_titanic_dataset():
    raw_data_path = os.path.join("data", "raw")
    os.makedirs(raw_data_path, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.competition_download_files("titanic", path=raw_data_path)

    print(f"Dataset downloaded to {raw_data_path}. Extracting...")
    unzip_command = f"unzip -o {os.path.join(raw_data_path, 'titanic.zip')} -d {raw_data_path}"
    os.system(unzip_command)
    print("Extraction complete.")

if __name__ == "__main__":
    download_titanic_dataset()
