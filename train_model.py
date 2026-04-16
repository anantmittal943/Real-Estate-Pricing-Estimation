from dataset_downloader import ensure_dataset_file
from ml_pipeline import train_and_save_model


def main() -> None:
    csv_path = ensure_dataset_file(force_download=False)
    metrics = train_and_save_model(csv_path)

    print("Model trained and saved successfully.")
    print(f"Dataset path: {csv_path}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"R2: {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
