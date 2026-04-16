from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
MODELS_DIR = ROOT_DIR / "models"

KAGGLE_DATASET = "harlfoxem/housesalesprediction"
DATASET_FILENAME = "kc_house_data.csv"
DATASET_CSV_PATH = DATASET_DIR / DATASET_FILENAME

MODEL_PATH = MODELS_DIR / "linear_regression_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

TEST_SIZE = 0.2
RANDOM_STATE = 42

REQUIRED_TRAIN_COLUMNS = [
    "id",
    "date",
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]

RAW_INFERENCE_COLUMNS = [
    "id",
    "date",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]
