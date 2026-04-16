from pathlib import Path
import shutil

import kagglehub

from config import DATASET_CSV_PATH, DATASET_DIR, KAGGLE_DATASET


def _copy_tree(source_dir: Path, target_dir: Path) -> None:
    for source_path in source_dir.rglob("*"):
        if source_path.is_file():
            relative_path = source_path.relative_to(source_dir)
            destination_path = target_dir / relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)


def find_dataset_csv(target_dir: Path = DATASET_DIR) -> Path:
    preferred_csv = target_dir / DATASET_CSV_PATH.name
    if preferred_csv.exists():
        return preferred_csv

    csv_files = sorted(target_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file was found in {target_dir}. Please run the downloader first."
        )

    return csv_files[0]


def download_dataset(target_dir: Path = DATASET_DIR) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    cached_dataset_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))

    _copy_tree(cached_dataset_path, target_dir)
    return find_dataset_csv(target_dir)


def ensure_dataset_file(force_download: bool = False) -> Path:
    if not force_download:
        try:
            return find_dataset_csv(DATASET_DIR)
        except FileNotFoundError:
            pass

    return download_dataset(DATASET_DIR)
