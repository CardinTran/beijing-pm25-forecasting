"""Train XGBoost on the PM2.5 data."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from random_forest import FEATURES, TARGET, add_weather_history, make_preprocessor, score
from src.data_loader import load_dataset
from src.preprocessing import prepare_data

# Settings to try. Each one is scored on 2013
PARAM_GRID = {
    # Poisson and Tweedie losses suit a skewed target that is never negative
    "objective": ["reg:squarederror", "count:poisson", "reg:tweedie"],
    "max_depth": [3, 4, 6],
    "min_child_weight": [1, 10, 50],
    # Train each tree on a random share of rows and features
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.5, 0.8],
}
# Stop adding trees once 2013 RMSE stops improving
MAX_TREES = 3000
EARLY_STOPPING_ROUNDS = 100


def make_regressor(n_estimators=MAX_TREES, **params):
    # Fix the seed for repeatable fits
    return XGBRegressor(
        n_estimators=n_estimators, learning_rate=0.03, tree_method="hist",
        eval_metric="rmse", random_state=42, n_jobs=-1, **params,
    )


def main():
    root = Path(__file__).resolve().parent
    data = load_dataset(root / "data" / "PRSA_data_2010.1.1-2014.12.31.csv")
    training, validation, testing = prepare_data(add_weather_history(data))

    # Fit the preprocessor on 2010–2012 only, then encode 2013 with it
    preprocessor = make_preprocessor().fit(training[FEATURES])
    train_x = preprocessor.transform(training[FEATURES])
    valid_x = preprocessor.transform(validation[FEATURES])

    # Train on 2010–2012 and score 2013 for every setting. Never look at 2014
    settings = list(ParameterGrid(PARAM_GRID))
    print(f"\nGrid search over {len(settings)} settings...")
    rows = []
    for params in settings:
        regressor = make_regressor(early_stopping_rounds=EARLY_STOPPING_ROUNDS, **params)
        regressor.fit(
            train_x, training[TARGET],
            eval_set=[(valid_x, validation[TARGET])], verbose=False,
        )
        rows.append({
            **params,
            "n_estimators": regressor.best_iteration + 1,
            "validation_RMSE": regressor.best_score,
        })
    results = pd.DataFrame(rows).sort_values("validation_RMSE")
    print("\nTop 10 settings by 2013 RMSE:")
    print(results.head(10).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    output_dir = root / "results"
    output_dir.mkdir(exist_ok=True)
    results.to_csv(output_dir / "xgboost_grid_search.csv", index=False)

    best = results.iloc[0].to_dict()
    best.pop("validation_RMSE")
    best["n_estimators"] = int(best["n_estimators"])
    best["max_depth"] = int(best["max_depth"])
    print(f"\nBest settings: {best}")

    # Refit the best setting on 2010–2013 with the tree count chosen on 2013
    print("\nBest settings refit on 2010–2013 and scored on 2014.")
    development = pd.concat([training, validation])
    model = Pipeline([("preprocessor", make_preprocessor()), ("xgboost", make_regressor(**best))])
    model.fit(development[FEATURES], development[TARGET])
    score(model, testing, "Test")

    # Gain is how much each feature's splits reduce the loss
    names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = pd.Series(model.named_steps["xgboost"].feature_importances_, index=names)
    print("\nFeature importances (share of total gain):")
    print(importances.sort_values(ascending=False).head(15).to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
