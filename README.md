# Real Estate Price Estimation

This project trains a baseline Linear Regression model for house price estimation and provides a Streamlit dashboard for interactive predictions.

## 1) Kaggle credential setup (required once)

1. Create a Kaggle account and generate API credentials.
2. Download `kaggle.json`.
3. Place it at:

   `%USERPROFILE%/.kaggle/kaggle.json`

## 2) Install dependencies

```bash
pip install -r requirements.txt
```

## 3) Download dataset and train model

```bash
python train_model.py
```

This command downloads the Kaggle dataset (`harlfoxem/housesalesprediction`) into the local `dataset` folder if needed, then trains and saves artifacts into `models`.

## 4) Launch dashboard

```bash
streamlit run dashboard_app.py
```

## Files

- `dataset_downloader.py`: handles Kaggle download and dataset discovery
- `ml_pipeline.py`: feature engineering, training, metrics, model persistence
- `train_model.py`: training entrypoint
- `dashboard_app.py`: Streamlit UI dashboard
- `real_estate_price_estimation_ml_project.py`: legacy-compatible local entrypoint
