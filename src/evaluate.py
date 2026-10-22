"""Score models and save plots."""

from pathlib import Path

import matplotlib

# Save plots without opening a window
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing import FEATURES, NUMERIC_FEATURES, TARGET


def evaluate_models(models, data, split):
    rows = []
    predictions = {}
    # Compare models on the same rows
    for name, model in models.items():
        predicted = model.predict(data[FEATURES])
        predictions[name] = predicted
        rows.append({
            "split": split,
            "model": name,
            "MAE": mean_absolute_error(data[TARGET], predicted),
            # RMSE gives more weight to large errors
            "RMSE": np.sqrt(mean_squared_error(data[TARGET], predicted)),
            # R² uses the evaluation mean as its reference
            "R2": r2_score(data[TARGET], predicted),
            "n_samples": len(data),
        })
    return pd.DataFrame(rows), predictions


def save_figures(metrics, predictions, testing, development, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    final_metrics = metrics.loc[metrics["split"] == "test"]

    # Separate metrics with different scales
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        axis.bar(final_metrics["model"], final_metrics[metric], color="steelblue")
        axis.set_title(metric if metric != "R2" else "R²")
        axis.set_ylabel("µg/m³" if metric != "R2" else "R² (dimensionless)")
        axis.tick_params(axis="x", labelrotation=35)
        axis.axhline(0, color="black", linewidth=0.6)
    fig.suptitle("Same-hour PM2.5 model comparison — held-out 2014")
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
    actual = testing[TARGET].to_numpy()
    # Use matching axes across models
    lower = min(0, actual.min(), min(values.min() for values in predictions.values()))
    upper = max(actual.max(), max(values.max() for values in predictions.values()))
    for axis, (name, predicted) in zip(axes.flat, predictions.items()):
        axis.scatter(actual, predicted, s=7, alpha=0.2, color="steelblue")
        # The diagonal marks perfect predictions
        axis.plot([lower, upper], [lower, upper], "--", color="black", linewidth=1)
        axis.set(title=name, xlabel="Actual PM2.5 (µg/m³)", ylabel="Predicted PM2.5 (µg/m³)")
        axis.set_aspect("equal", adjustable="box")
    fig.suptitle("Actual vs. predicted same-hour PM2.5 — held-out 2014")
    fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_predicted.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Keep 2014 out of exploratory plots
    correlation = development[NUMERIC_FEATURES + [TARGET]].corr()
    fig, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axis)
    axis.set_title("Pearson correlations — 2010–2013")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)