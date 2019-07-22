import numpy as np
import pandas as pd

from src.preprocess import (
    FEATURE_COLS,
    median_impute,
    prepare_xy,
    replace_zeros_with_nan,
)


def make_df():
    return pd.DataFrame({
        "Pregnancies": [1, 2, 3, 4],
        "Glucose": [120, 0, 140, 100],
        "BloodPressure": [70, 80, 0, 0],
        "SkinThickness": [20, 0, 25, 30],
        "Insulin": [80, 0, 0, 90],
        "BMI": [25.0, 30.0, 0.0, 28.0],
        "DiabetesPedigreeFunction": [0.5, 0.6, 0.7, 0.8],
        "Age": [33, 40, 50, 22],
        "Outcome": [1, 0, 1, 0],
    })


def test_replace_zeros_with_nan_only_in_listed_cols():
    df = make_df()
    out = replace_zeros_with_nan(df, ["Glucose", "BMI"])
    # Glucose 0 -> NaN
    assert np.isnan(out.loc[1, "Glucose"])
    # BMI 0 -> NaN
    assert np.isnan(out.loc[2, "BMI"])
    # SkinThickness 0 untouched because not in the list
    assert out.loc[1, "SkinThickness"] == 0


def test_median_impute_replaces_nan():
    df = make_df()
    df = replace_zeros_with_nan(df, ["Glucose"])
    out = median_impute(df, ["Glucose"])
    # imputed value is the median of the non-missing column.
    expected = df["Glucose"].median()
    assert out.loc[1, "Glucose"] == expected
    assert not out["Glucose"].isnull().any()


def test_prepare_xy_shapes_and_order():
    df = make_df()
    X, y = prepare_xy(df)
    assert X.shape == (4, len(FEATURE_COLS))
    assert y.shape == (4,)
    # first column should be Pregnancies
    assert list(X[:, 0]) == [1, 2, 3, 4]
