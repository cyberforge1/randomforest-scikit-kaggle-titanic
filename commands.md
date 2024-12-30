# Commands

## VENV

python3.11 -m venv venv

source venv/bin/activate

pip install -r requirements.txt


## Download the Dataset
python src/data/download.py


## Data Preprocessing
python src/data/preprocess.py


## Train the Model
python -m src.models.train


## Interact with the Model (CLI)
python main.py --input "age=22,sex=female,class=3"


## Run the Entire Pipeline
# Automate all steps: dataset download, preprocessing, and training
python src/pipeline.py


## Run Tests
python -m unittest discover tests
