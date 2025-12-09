# src/inference.py

from pathlib import Path
import joblib
import pandas as pd


def load_model():
    """
    Kaydedilmiş final modeli yükler.
    """
    root_dir = Path(__file__).resolve().parents[1]
    model_path = root_dir / "models" / "final_model.pkl"
    model = joblib.load(model_path)
    return model


def predict_diabetes_risk(input_dict: dict) -> float:
    """
    Streamlit veya başka bir yerden gelen feature sözlüğünü alır,
    DataFrame'e çevirir ve diyabet risk olasılığını döner.
    """
    model = load_model()

    # input_dict → tek satırlık DataFrame (feature isimleri aynı olmalı!)
    df_input = pd.DataFrame([input_dict])

    prob = model.predict_proba(df_input)[0][1]
    return prob
