"""Fit models and tune them on 2013 data."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.preprocessing import EXPANDED_FEATURES, TARGET, make_preprocessor

SEED = 42
ORIGINAL_FOREST = "Original Random Forest"
TUNED_FOREST = "Tuned Random Forest"
HIST_BOOSTING = "Histogram Gradient Boosting"
XGBOOST = "XGBoost"

GRIDS = {
    TUNED_FOREST: {
        "n_estimators": [100, 300],
        "max_depth": [None, 20],
        "min_samples_leaf": [5, 10, 20, 50],
        "max_features": [1.0, 0.5, 0.33],
    },
    HIST_BOOSTING: {
        "loss": ["squared_error", "poisson"],
        "learning_rate": [0.03, 0.1],
        "max_iter": [300, 1000],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [20, 100],
    },
    XGBOOST: {
        "objective": ["reg:squarederror", "count:poisson", "reg:tweedie"],
        "max_depth": [3, 4, 6],
        "min_child_weight": [1, 10, 50],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.5, 0.8],
    },
}
GRID_FILES = {
    TUNED_FOREST: "random_forest_grid_search.csv",
    HIST_BOOSTING: "gradient_boosting_grid_search.csv",
    XGBOOST: "xgboost_grid_search.csv",
}
# Keep the column names used in the original search files.
GRID_PREFIXES = {
    TUNED_FOREST: "param_forest__",
    HIST_BOOSTING: "param_boosting__",
}
# The forest and histogram models use their earlier settings as controls.
CONTROL_SETTINGS = {TUNED_FOREST: {}, HIST_BOOSTING: {}, XGBOOST: {"n_estimators": 100}}


def make_regressor(name, jobs, params):
    if name == "Mean baseline":
        return DummyRegressor(strategy="mean")
    if name == "Linear Regression":
        return LinearRegression()
    if name == "Ridge Regression":
        return Ridge(alpha=1.0)
    if name == ORIGINAL_FOREST:
        return RandomForestRegressor(
            n_estimators=100, min_samples_leaf=2, random_state=SEED, n_jobs=jobs,
        )
    if name == TUNED_FOREST:
        settings = {"n_estimators": 200, "min_samples_leaf": 2, **params}
        return RandomForestRegressor(random_state=SEED, n_jobs=jobs, **settings)
    if name == HIST_BOOSTING:
        return HistGradientBoostingRegressor(early_stopping=False, random_state=SEED, **params)
    if name == XGBOOST:
        settings = {
            "n_estimators": 3000, "learning_rate": 0.03, "tree_method": "hist",
            "eval_metric": "rmse", "random_state": SEED, "n_jobs": jobs, **params,
        }
        if settings.get("objective") == "reg:tweedie":
            settings["tweedie_variance_power"] = 1.5
        return XGBRegressor(**settings)
    raise ValueError(f"Unknown model: {name}")


def fit_model(name, data, features, jobs, params):
    scale = name in {"Mean baseline", "Linear Regression", "Ridge Regression", ORIGINAL_FOREST}
    model = Pipeline([
        ("preprocessor", make_preprocessor(features, scale=scale)),
        ("regressor", make_regressor(name, jobs, params)),
    ])
    return model.fit(data[features], data[TARGET])


def tune_model(name, training, validation, jobs):
    features = EXPANDED_FEATURES
    preprocessor = make_preprocessor(features).fit(training[features])
    train_x = preprocessor.transform(training[features])
    valid_x = preprocessor.transform(validation[features])
    train_y, valid_y = training[TARGET], validation[TARGET]

    rows = []
    grid = ParameterGrid(GRIDS[name])
    print(f"Trying {len(grid)} settings for {name}", flush=True)
    for params in grid:
        if name == XGBOOST:
            regressor = make_regressor(name, jobs, {**params, "early_stopping_rounds": 100})
            regressor.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose=False)
            params = {**params, "n_estimators": regressor.best_iteration + 1}
        else:
            regressor = make_regressor(name, jobs, params).fit(train_x, train_y)
        rmse = root_mean_squared_error(valid_y, regressor.predict(valid_x))
        rows.append({**params, "validation_RMSE": rmse})

    best = min(rows, key=lambda row: row["validation_RMSE"])
    results = pd.DataFrame(rows).sort_values("validation_RMSE", kind="stable")
    if not np.isfinite(results["validation_RMSE"]).all():
        raise ValueError(f"{name} produced nonfinite validation scores.")
    return {key: value for key, value in best.items() if key != "validation_RMSE"}, results
