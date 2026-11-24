"""Run the PM2.5 experiments."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from src.data_loader import load_dataset
from src.evaluate import analyze_test, evaluate_model, save_figures
from src.preprocessing import EXPANDED_FEATURES, FEATURES, TARGET, prepare_data
from src.train import (
    CONTROL_SETTINGS, GRID_FILES, GRID_PREFIXES, GRIDS, ORIGINAL_FOREST, fit_model, tune_model,
)

ROOT = Path(__file__).resolve().parent


def run_experiment(args):
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    data = load_dataset(ROOT / "data/PRSA_data_2010.1.1-2014.12.31.csv")
    training, validation, testing = prepare_data(data)
    development = pd.concat([training, validation])

    settings_file = ROOT / "results/best_hyperparameters.json"
    if args.mode == "full":
        selected = {}
        for name, filename in GRID_FILES.items():
            selected[name], search = tune_model(name, training, validation, args.jobs)
            if name in GRID_PREFIXES:
                search = search[[*GRIDS[name], "validation_RMSE"]].rename(
                    columns={key: GRID_PREFIXES[name] + key for key in GRIDS[name]},
                )
            search.to_csv(output / filename, index=False)
    else:
        print("Using saved model settings", flush=True)
        selected = json.loads(settings_file.read_text())
    (output / settings_file.name).write_text(json.dumps(selected, indent=2) + "\n")

    base_models = ["Mean baseline", "Linear Regression", "Ridge Regression", ORIGINAL_FOREST]
    configurations = [(name, "original", FEATURES, {}) for name in base_models]
    for name in GRID_FILES:
        params = selected[name]
        configurations.extend([
            (name, "tuned", EXPANDED_FEATURES, params),
            (name, "tuned", FEATURES, params),
            (name, "untuned", EXPANDED_FEATURES, CONTROL_SETTINGS[name]),
            (name, "untuned", FEATURES, CONTROL_SETTINGS[name]),
        ])

    comparison, ablations, final_models = [], [], {}
    predictions = {
        "validation": validation[[TARGET, "season"]].rename(columns={TARGET: "actual"}),
        "test": testing[[TARGET, "season"]].rename(columns={TARGET: "actual"}),
    }
    splits = [("validation", training, validation), ("test", development, testing)]
    for name, setting, features, params in configurations:
        compare = setting == "original" or (setting == "tuned" and features == EXPANDED_FEATURES)
        print(f"Fitting {name} with {setting} settings and {len(features)} features", flush=True)
        for split, fit_data, score_data in splits:
            model = fit_model(name, fit_data, features, args.jobs, params)
            metrics, predicted = evaluate_model(model, score_data, features)
            row = {"split": split, "model": name, **metrics}
            if setting != "original":
                ablations.append({**row, "settings": setting, "n_features": len(features)})
            if compare:
                comparison.append(row)
                predictions[split][name] = predicted
                if split == "test":
                    final_models[name] = (model, features)

    metrics = pd.DataFrame(comparison).sort_values("split", ascending=False, kind="stable")
    ablation = pd.DataFrame(ablations)
    metrics.to_csv(output / "model_comparison.csv", index=False)
    ablation.to_csv(output / "feature_ablation.csv", index=False)
    pd.concat(predictions, names=["split"]).to_csv(output / "predictions.csv")
    importance, pollution, seasonal = analyze_test(
        final_models, predictions["test"], testing, development, output,
    )
    figure_dir = ROOT / "report/figures" if output.resolve() == ROOT / "results" else output / "figures"
    save_figures(
        metrics, predictions["test"], ablation, importance, pollution, seasonal, figure_dir,
    )
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["full", "saved"], default="saved",
                        help="run the full search or use saved settings")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1),
                        help="worker threads per model")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    with threadpool_limits(limits=1, user_api="blas"):
        with threadpool_limits(limits=args.jobs, user_api="openmp"):
            run_experiment(args)


if __name__ == "__main__":
    main()
