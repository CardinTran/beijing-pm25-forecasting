"""Train a Random Forest on the PM2.5 data."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_loader import load_dataset
from src.preprocessing import prepare_data

# Wind direction and season have no order, so one-hot encode them
CATEGORICAL_FEATURES = ["cbwd", "season"]
# Pollution builds up over days, so describe recent weather too
HISTORY_COLUMNS = ["DEWP", "PRES", "TEMP", "Iws"]
HISTORY_WINDOWS = [24, 72]
HISTORY_FEATURES = (
    [f"{column}_change_3h" for column in HISTORY_COLUMNS]
    + [f"{column}_mean_{hours}h" for column in HISTORY_COLUMNS for hours in HISTORY_WINDOWS]
    + ["Ir_total_24h"]
)
NUMERIC_FEATURES = ["TEMP", "DEWP", "PRES", "Iws", "Is", "Ir", "hour", "month"] + HISTORY_FEATURES
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "pm2.5"

# Settings to try. Each one is scored on 2013
PARAM_GRID = {
    "forest__n_estimators": [100, 300],
    "forest__max_depth": [None, 20],
    "forest__min_samples_leaf": [5, 10, 20, 50],
    "forest__max_features": [1.0, 0.5, 0.33],
}


def add_weather_history(data):
    # Use the full hourly series, including hours with missing PM2.5
    dates = pd.to_datetime(data[["year", "month", "day", "hour"]])
    data = data.assign(timestamp=dates).sort_values("timestamp")
    if not (data["timestamp"].diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("Weather history needs one row per hour with no gaps.")
    # Only past and current weather is used, never PM2.5
    for column in HISTORY_COLUMNS:
        data[f"{column}_change_3h"] = data[column].diff(3)
        for hours in HISTORY_WINDOWS:
            data[f"{column}_mean_{hours}h"] = data[column].rolling(hours, min_periods=1).mean()
    data["Ir_total_24h"] = data["Ir"].rolling(24, min_periods=1).sum()
    return data.drop(columns="timestamp")


def make_preprocessor():
    # Trees split on thresholds, so numeric features need no scaling
    numeric = SimpleImputer(strategy="median")
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ], verbose_feature_names_out=False)


def make_model():
    preprocessor = make_preprocessor()
    # Fix the seed for repeatable fits
    forest = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("forest", forest)])


def score(model, data, split):
    predicted = model.predict(data[FEATURES])
    actual = data[TARGET]
    print(
        f"{split}: MAE {mean_absolute_error(actual, predicted):.2f} µg/m³ | "
        f"RMSE {np.sqrt(mean_squared_error(actual, predicted)):.2f} µg/m³ | "
        f"R² {r2_score(actual, predicted):.3f}"
    )


def main():
    root = Path(__file__).resolve().parent
    data = load_dataset(root / "data" / "PRSA_data_2010.1.1-2014.12.31.csv")
    training, validation, testing = prepare_data(add_weather_history(data))

    print("\nTrain on 2010–2012 and score 2013 with the default settings.")
    model = make_model().fit(training[FEATURES], training[TARGET])
    score(model, validation, "Validation")

    # Train on 2010–2012 and score 2013 for every setting. Never look at 2014
    development = pd.concat([training, validation])
    folds = PredefinedSplit(np.r_[np.full(len(training), -1), np.zeros(len(validation))])
    search = GridSearchCV(
        make_model(), PARAM_GRID, cv=folds, scoring="neg_root_mean_squared_error",
        # Refit the best setting on 2010–2013
        refit=True, verbose=1,
    )
    total = np.prod([len(values) for values in PARAM_GRID.values()])
    print(f"\nGrid search over {total} settings...")
    search.fit(development[FEATURES], development[TARGET])

    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    results["validation_RMSE"] = -results["mean_test_score"]
    columns = [f"param_{name}" for name in PARAM_GRID] + ["validation_RMSE"]
    print("\nTop 10 settings by 2013 RMSE:")
    print(results[columns].head(10).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nBest settings: {search.best_params_}")
    output_dir = root / "results"
    output_dir.mkdir(exist_ok=True)
    results[columns].to_csv(output_dir / "random_forest_grid_search.csv", index=False)

    print("\nBest settings refit on 2010–2013 and scored on 2014.")
    model = search.best_estimator_
    score(model, testing, "Test")

    # Show the encoded columns the forest actually used
    names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = pd.Series(model.named_steps["forest"].feature_importances_, index=names)
    print("\nFeature importances:")
    print(importances.sort_values(ascending=False).to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
