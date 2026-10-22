"""Run the model comparison."""

from pathlib import Path

import pandas as pd

from src.data_loader import load_dataset
from src.evaluate import evaluate_models, save_figures
from src.preprocessing import prepare_data
from src.train import train_models


def main():
    # Use paths relative to this file
    root = Path(__file__).resolve().parent
    dataset = root / "data" / "PRSA_data_2010.1.1-2014.12.31.csv"
    data = load_dataset(dataset)
    print(f"Loaded {len(data):,} observations.\n")
    training, validation, testing = prepare_data(data)
    development = pd.concat([training, validation])

    print("\nValidation: train on 2010–2012 and score 2013.")
    # Validate on a later year
    validation_models = train_models(training)
    validation_metrics, _ = evaluate_models(validation_models, validation, "validation")

    print("\nTest: refit on 2010–2013 and score 2014.")
    # Refit on 2010–2013 before testing on 2014
    final_models = train_models(development)
    test_metrics, predictions = evaluate_models(final_models, testing, "test")
    metrics = pd.concat([validation_metrics, test_metrics], ignore_index=True)
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    metrics.to_csv(results_dir / "model_comparison.csv", index=False)
    save_figures(metrics, predictions, testing, development, results_dir / "figures")

    print("\nModel comparison (MAE and RMSE in µg/m³):")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved results to {results_dir}")

if __name__ == "__main__":
    main()
