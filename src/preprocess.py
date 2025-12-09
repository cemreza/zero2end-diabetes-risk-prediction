# src/preprocess.py

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(sample_frac: float = 0.10, random_state: int = 42) -> pd.DataFrame:
    """
    Kaggle diyabet veri setini yükler.
    sample_frac: 0.10 → verinin %10'u ile çalış (eğitimi hızlandırmak için)
    """
    root_dir = Path(__file__).resolve().parents[1]
    data_path = root_dir / "data" / "raw" / "diabetes_012_health_indicators_BRFSS2015.csv"

    df = pd.read_csv(data_path)

    if sample_frac is not None and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=random_state)

    return df


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Diabetes_012 → 0 (sağlıklı), 1 ve 2 (riskli) olacak şekilde
    binary hedef değişken ekler.
    """
    df = df.copy()
    df["Diabetes_binary"] = df["Diabetes_012"].apply(lambda x: 0 if x == 0 else 1)
    return df


def get_features_and_target(df: pd.DataFrame):
    """
    X (özellikler) ve y (hedef) olarak ayırır.
    """
    X = df.drop(["Diabetes_012", "Diabetes_binary"], axis=1)
    y = df["Diabetes_binary"]
    return X, y


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Eğitim ve test setlerine böler (stratified).
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
