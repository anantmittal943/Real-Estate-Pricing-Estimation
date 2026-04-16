from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from config import RAW_INFERENCE_COLUMNS
from dataset_downloader import ensure_dataset_file
from ml_pipeline import (
    build_inference_frame,
    get_test_predictions,
    load_model_artifacts,
    load_raw_data,
    train_and_save_model,
)


st.set_page_config(
    page_title="Real Estate Price Dashboard",
    page_icon="🏠",
    layout="wide",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif;
        }

        .stApp {
            background: radial-gradient(circle at top left, #f9f7f2, #e6edf4 45%, #dde3e9 100%);
        }

        .stMetric {
            background: rgba(255, 255, 255, 0.65);
            border-radius: 12px;
            padding: 10px;
            border: 1px solid #d2dae3;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def _compute_default_date(dataframe: pd.DataFrame):
    parsed_dates = pd.to_datetime(dataframe["date"], errors="coerce").dropna()
    if parsed_dates.empty:
        return pd.Timestamp("2014-01-01").date()
    return parsed_dates.iloc[0].date()


def _default_numeric_value(dataframe: pd.DataFrame, column_name: str) -> float:
    if column_name not in dataframe.columns:
        return 0.0
    series = pd.to_numeric(dataframe[column_name], errors="coerce").dropna()
    if series.empty:
        return 0.0
    return float(series.median())


@st.cache_data
def get_dataset() -> pd.DataFrame:
    csv_path = ensure_dataset_file(force_download=False)
    return load_raw_data(Path(csv_path))


@st.cache_resource
def get_model_and_metadata():
    try:
        return load_model_artifacts()
    except FileNotFoundError:
        csv_path = ensure_dataset_file(force_download=False)
        train_and_save_model(Path(csv_path))
        return load_model_artifacts()


def render_prediction_page(dataframe: pd.DataFrame, model, metadata) -> None:
    st.subheader("Predict Property Price")

    default_date = _compute_default_date(dataframe)
    sale_date = st.date_input("Sale Date", value=default_date)

    col1, col2, col3 = st.columns(3)

    with col1:
        bedrooms = st.number_input("Bedrooms", min_value=0.0, value=_default_numeric_value(dataframe, "bedrooms"), step=1.0)
        bathrooms = st.number_input("Bathrooms", min_value=0.0, value=_default_numeric_value(dataframe, "bathrooms"), step=0.25)
        sqft_living = st.number_input("Sqft Living", min_value=100, value=int(_default_numeric_value(dataframe, "sqft_living")), step=10)
        sqft_lot = st.number_input("Sqft Lot", min_value=300, value=int(_default_numeric_value(dataframe, "sqft_lot")), step=100)
        floors = st.number_input("Floors", min_value=1.0, value=_default_numeric_value(dataframe, "floors"), step=0.5)
        grade = st.number_input("Grade", min_value=1, max_value=13, value=int(_default_numeric_value(dataframe, "grade")), step=1)

    with col2:
        condition = st.number_input("Condition", min_value=1, max_value=5, value=int(_default_numeric_value(dataframe, "condition")), step=1)
        view = st.number_input("View", min_value=0, max_value=4, value=int(_default_numeric_value(dataframe, "view")), step=1)
        waterfront = st.number_input("Waterfront", min_value=0, max_value=1, value=int(_default_numeric_value(dataframe, "waterfront")), step=1)
        sqft_above = st.number_input("Sqft Above", min_value=100, value=int(_default_numeric_value(dataframe, "sqft_above")), step=10)
        sqft_basement = st.number_input("Sqft Basement", min_value=0, value=int(_default_numeric_value(dataframe, "sqft_basement")), step=10)
        sqft_living15 = st.number_input("Sqft Living (Neighborhood)", min_value=100, value=int(_default_numeric_value(dataframe, "sqft_living15")), step=10)

    with col3:
        sqft_lot15 = st.number_input("Sqft Lot (Neighborhood)", min_value=300, value=int(_default_numeric_value(dataframe, "sqft_lot15")), step=100)
        yr_built = st.number_input("Year Built", min_value=1900, max_value=2026, value=int(_default_numeric_value(dataframe, "yr_built")), step=1)
        yr_renovated = st.number_input("Year Renovated", min_value=0, max_value=2026, value=int(_default_numeric_value(dataframe, "yr_renovated")), step=1)
        zipcode = st.number_input("Zipcode", min_value=10000, value=int(_default_numeric_value(dataframe, "zipcode")), step=1)
        latitude = st.number_input("Latitude", value=_default_numeric_value(dataframe, "lat"), step=0.001, format="%.5f")
        longitude = st.number_input("Longitude", value=_default_numeric_value(dataframe, "long"), step=0.001, format="%.5f")

    if st.button("Estimate Price", type="primary"):
        raw_input = {
            "id": 0,
            "date": sale_date.isoformat(),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft_living": sqft_living,
            "sqft_lot": sqft_lot,
            "floors": floors,
            "waterfront": waterfront,
            "view": view,
            "condition": condition,
            "grade": grade,
            "sqft_above": sqft_above,
            "sqft_basement": sqft_basement,
            "yr_built": yr_built,
            "yr_renovated": yr_renovated,
            "zipcode": zipcode,
            "lat": latitude,
            "long": longitude,
            "sqft_living15": sqft_living15,
            "sqft_lot15": sqft_lot15,
        }

        missing_columns = [col for col in RAW_INFERENCE_COLUMNS if col not in raw_input]
        if missing_columns:
            st.error("Missing input columns: " + ", ".join(missing_columns))
            return

        inference_frame = build_inference_frame(raw_input, metadata["feature_columns"])
        prediction = float(model.predict(inference_frame)[0])

        st.metric("Estimated Sale Price", f"${prediction:,.0f}")


def render_data_explorer_page(dataframe: pd.DataFrame) -> None:
    st.subheader("Data Explorer")
    st.write(f"Rows: {len(dataframe):,} | Columns: {dataframe.shape[1]}")
    st.dataframe(dataframe.head(25), use_container_width=True)

    fig_price, ax_price = plt.subplots(figsize=(9, 5))
    sns.histplot(dataframe["price"], bins=40, kde=True, ax=ax_price)
    ax_price.set_title("Distribution of House Prices")
    ax_price.set_xlabel("Price")
    st.pyplot(fig_price)

    numeric_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
    correlation_matrix = dataframe[numeric_columns].corr(numeric_only=True)

    fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, cmap="coolwarm", center=0, ax=ax_corr)
    ax_corr.set_title("Correlation Heatmap")
    st.pyplot(fig_corr)


def render_model_page(dataframe: pd.DataFrame, model, metadata) -> None:
    st.subheader("Model Performance")

    model_metrics = metadata.get("metrics", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{model_metrics.get('mae', 0):,.2f}")
    col2.metric("MSE", f"{model_metrics.get('mse', 0):,.2f}")
    col3.metric("R2", f"{model_metrics.get('r2', 0):.4f}")

    predictions, refreshed_metrics = get_test_predictions(model, dataframe)

    st.caption(
        "Metrics above come from saved training metadata. "
        f"Current recalculated R2 on holdout split: {refreshed_metrics['r2']:.4f}"
    )

    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=predictions["actual_price"],
        y=predictions["predicted_price"],
        alpha=0.6,
        ax=ax_scatter,
    )
    min_price = predictions["actual_price"].min()
    max_price = predictions["actual_price"].max()
    ax_scatter.plot([min_price, max_price], [min_price, max_price], "r--", linewidth=2)
    ax_scatter.set_title("Actual vs Predicted Prices")
    ax_scatter.set_xlabel("Actual Price")
    ax_scatter.set_ylabel("Predicted Price")
    st.pyplot(fig_scatter)

    residuals = predictions["actual_price"] - predictions["predicted_price"]
    fig_residual, ax_residual = plt.subplots(figsize=(8, 4))
    sns.histplot(residuals, bins=40, kde=True, ax=ax_residual)
    ax_residual.set_title("Residual Distribution")
    ax_residual.set_xlabel("Actual - Predicted")
    st.pyplot(fig_residual)


def main() -> None:
    st.title("Real Estate Pricing Intelligence Dashboard")
    st.write("A practical dashboard for price estimation, data exploration, and baseline model diagnostics.")

    try:
        dataframe = get_dataset()
        model, metadata = get_model_and_metadata()
    except Exception as exc:
        st.error(str(exc))
        st.info(
            "If dataset download fails, add your kaggle.json to %USERPROFILE%/.kaggle and rerun."
        )
        return

    selected_page = st.sidebar.radio(
        "Navigate",
        ("Predict Price", "Data Explorer", "Model Performance"),
    )

    if selected_page == "Predict Price":
        render_prediction_page(dataframe, model, metadata)
    elif selected_page == "Data Explorer":
        render_data_explorer_page(dataframe)
    else:
        render_model_page(dataframe, model, metadata)


if __name__ == "__main__":
    main()
