"""Prepare features and split by year."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["TEMP", "DEWP", "PRES", "Iws", "Is", "Ir", "hour", "month"]
CATEGORICAL_FEATURES = ["cbwd", "season"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "pm2.5"
HISTORY_COLUMNS = ["DEWP", "PRES", "TEMP", "Iws"]
HISTORY_WINDOWS = [24, 72]
HISTORY_FEATURES = (
    [f"{column}_change_3h" for column in HISTORY_COLUMNS]
    + [f"{column}_mean_{hours}h" for column in HISTORY_COLUMNS for hours in HISTORY_WINDOWS]
    + ["Ir_total_24h"]
)
EXPANDED_FEATURES = NUMERIC_FEATURES + HISTORY_FEATURES + CATEGORICAL_FEATURES


def add_weather_history(data):
    if not (data.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("Weather history needs one row per hour with no gaps.")
    for column in HISTORY_COLUMNS:
        data[f"{column}_change_3h"] = data[column].diff(3)
        for hours in HISTORY_WINDOWS:
            data[f"{column}_mean_{hours}h"] = data[column].rolling(hours, min_periods=1).mean()
    data["Ir_total_24h"] = data["Ir"].rolling(24, min_periods=1).sum()
    return data


def prepare_data(data):
    data = data.copy()
    print("Missing values by column")
    print(data.isna().sum().to_string())

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
    data.index = pd.DatetimeIndex(dates, name="timestamp")
    data = data.sort_index()

    # Build weather history before dropping hours with missing PM2.5.
    data = add_weather_history(data)
    missing_targets = data[TARGET].isna().sum()
    data = data.dropna(subset=[TARGET]).copy()
    print(f"Removed {missing_targets:,} rows with missing PM2.5")
    seasons = {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn",
    }
    data["season"] = data["month"].map(seasons)
    data["cbwd"] = data["cbwd"].replace(r"^\s*$", np.nan, regex=True)

    training = data.loc[data.index.year <= 2012].copy()
    validation = data.loc[data.index.year == 2013].copy()
    testing = data.loc[data.index.year == 2014].copy()
    for name, split in [("Training", training), ("Validation", validation), ("Test", testing)]:
        if len(split) < 2:
            raise ValueError(f"{name} needs at least two rows with PM2.5 values.")
        print(f"{name} has {len(split):,} rows from {split.index.min()} to {split.index.max()}")
    return training, validation, testing


def make_preprocessor(features=FEATURES, scale=False):
    numeric_features = [column for column in features if column not in CATEGORICAL_FEATURES]
    numeric = SimpleImputer(strategy="median", keep_empty_features=True)
    if scale:
        numeric = Pipeline([("imputer", numeric), ("scaler", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, numeric_features),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
