"""Train the four models."""

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from src.preprocessing import FEATURES, TARGET, make_preprocessor


def train_models(data):
    # Fix settings before checking scores
    regressors = {
        # Use the training mean as a baseline
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        # Ridge penalizes large coefficients
        "Ridge Regression": Ridge(alpha=1.0),
        # Trees capture nonlinear patterns. Fix the seed for repeatable fits
        "Random Forest": RandomForestRegressor(
            n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1,
        ),
    }
    models = {}
    for name, regressor in regressors.items():
        print(f"Training {name}...")
        # Fit a separate preprocessor for each model
        model = Pipeline([
            ("preprocessor", make_preprocessor()),
            ("regressor", regressor),
        ])
        model.fit(data[FEATURES], data[TARGET])
        models[name] = model
    return models