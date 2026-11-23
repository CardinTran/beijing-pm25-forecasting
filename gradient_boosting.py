"""Train gradient-boosted trees on the PM2.5 data."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline

from random_forest import FEATURES, TARGET, add_weather_history, make_preprocessor, score
from src.data_loader import load_dataset
from src.preprocessing import prepare_data

# Settings to try. Each one is scored on 2013
PARAM_GRID = {
    # Poisson loss suits a skewed target that is never negative
    "boosting__loss": ["squared_error", "poisson"],
    "boosting__learning_rate": [0.03, 0.1],
    "boosting__max_iter": [300, 1000],
    "boosting__max_leaf_nodes": [15, 31, 63],
    "boosting__min_samples_leaf": [20, 100],
}


def make_model():
    # Each tree corrects the errors of the trees before it. Fix the seed for repeatable fits
    boosting = HistGradientBoostingRegressor(early_stopping=False, random_state=42)
    return Pipeline([("preprocessor", make_preprocessor()), ("boosting", boosting)])


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
    results[columns].to_csv(output_dir / "gradient_boosting_grid_search.csv", index=False)

    print("\nBest settings refit on 2010–2013 and scored on 2014.")
    model = search.best_estimator_
    score(model, testing, "Test")

    # Boosting has no built-in importances. Measure how much RMSE rises when each feature is shuffled
    importances = permutation_importance(
        model, testing[FEATURES], testing[TARGET], scoring="neg_root_mean_squared_error",
        n_repeats=5, random_state=42, n_jobs=-1,
    )
    importances = pd.Series(importances.importances_mean, index=FEATURES)
    print("\nRMSE increase when each feature is shuffled (µg/m³):")
    print(importances.sort_values(ascending=False).to_string(float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
