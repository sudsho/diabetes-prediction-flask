"""Preprocessing for the Pima diabetes dataset."""
import numpy as np
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


def replace_zeros_with_nan(df, cols):
    """Several pima columns use 0 to mean missing. Convert those to NaN."""
    df = df.copy()
    for c in cols:
        df[c] = df[c].replace(0, np.nan)
    return df


def median_impute(df, cols):
    """Impute missing values in `cols` with the column median."""
    df = df.copy()
    for c in cols:
        med = df[c].median()
        df[c] = df[c].fillna(med)
    return df


def prepare_xy(df):
    """Split a clean df into feature matrix X and target vector y."""
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    return X, y
