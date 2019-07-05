"""Preprocessing for the Pima diabetes dataset."""
import pandas as pd


FEATURE_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET_COL = "Outcome"


def load_csv(path):
    df = pd.read_csv(path)
    return df
