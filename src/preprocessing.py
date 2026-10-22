"""Prepare features and split by year."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Use same-hour weather and time features
NUMERIC_FEATURES = ["TEMP", "DEWP", "PRES", "Iws", "Is", "Ir", "hour", "month"]
CATEGORICAL_FEATURES = ["cbwd", "season"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "pm2.5"


def prepare_data(data):
    # Keep the loaded data unchanged
    data = data.copy()
    print("Missing values:")
    print(data.isna().sum().to_string())

    # Catch invalid values before training
    columns = NUMERIC_FEATURES + [TARGET, "year", "day"]
    try:
        data[columns] = data[columns].apply(pd.to_numeric, errors="raise")
        dates = pd.to_datetime(data[["year", "month", "day", "hour"]])
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid numbers or timestamps.") from error
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Timestamps must be present and unique.")
    if np.isinf(data[columns].to_numpy(dtype=float)).any():
        raise ValueError("Infinite values found in the dataset.")
    if not data["year"].between(2010, 2014).all():
        raise ValueError("Expected data from 2010 through 2014.")
    # Sort by time before splitting
    data.index = pd.DatetimeIndex(dates, name="timestamp")
    data = data.sort_index()

    # Drop missing targets instead of guessing them
    missing_targets = data[TARGET].isna().sum()
    data = data.dropna(subset=[TARGET]).copy()
    print(f"\nRemoved {missing_targets:,} rows with missing PM2.5.")
    # Group months into seasons
    seasons = {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn",
    }
    data["season"] = data["month"].map(seasons)
    # Treat blank wind directions as missing
    data["cbwd"] = data["cbwd"].replace(r"^\s*$", np.nan, regex=True)

    # Save 2013 for validation and 2014 for testing
    training = data.loc[data.index.year <= 2012].copy()
    validation = data.loc[data.index.year == 2013].copy()
    testing = data.loc[data.index.year == 2014].copy()
    for name, split in [("Training", training), ("Validation", validation), ("Test", testing)]:
        # R² needs at least two rows.
        if len(split) < 2:
            raise ValueError(f"{name} needs at least two rows with PM2.5 values.")
        print(f"{name}: {len(split):,} rows | {split.index.min()} to {split.index.max()}")
    return training, validation, testing


def make_preprocessor():
    # Fill numeric gaps with the median. Scale for the linear models
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    # Fill category gaps and encode without implying an order
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    # Fit on training rows only. Exclude the target and unlisted columns
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])