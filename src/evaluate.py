"""Score models and make plots."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from src.preprocessing import TARGET
from src.train import GRID_FILES, HIST_BOOSTING, ORIGINAL_FOREST, SEED, TUNED_FOREST, XGBOOST

SEASONS = ["winter", "spring", "summer", "autumn"]
UNIT = "µg/m³"
MODEL_LABELS = {
    "Mean baseline": "Mean", "Linear Regression": "Linear", "Ridge Regression": "Ridge",
    ORIGINAL_FOREST: "Original RF", TUNED_FOREST: "Tuned RF",
    HIST_BOOSTING: "Hist. GB", XGBOOST: "XGBoost",
}


def score(actual, predicted):
    if not np.isfinite(predicted).all():
        raise ValueError("Model produced nonfinite predictions.")
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": root_mean_squared_error(actual, predicted),
        "R2": r2_score(actual, predicted),
        "n_samples": len(actual),
    }


def evaluate_model(model, data, features):
    prediction = np.asarray(model.predict(data[features]), dtype=float)
    return score(data[TARGET], prediction), prediction


def paired_bootstrap(predictions, models):
    # Keep missing hours so each block covers the same amount of time.
    hourly = predictions.asfreq("h")
    errors = hourly[models].sub(hourly["actual"], axis=0).to_numpy()
    valid = hourly["actual"].notna().to_numpy()
    squared = np.nan_to_num(errors ** 2)
    # Running totals let us count observations and errors in each block.
    counts = np.r_[0, valid.cumsum()]
    sums = np.vstack([np.zeros(len(models)), squared.cumsum(axis=0)])
    comparisons = [(TUNED_FOREST, ORIGINAL_FOREST), (HIST_BOOSTING, TUNED_FOREST),
                   (XGBOOST, HIST_BOOSTING)]
    rng = np.random.default_rng(SEED)
    rows = []
    repetitions = 2000
    observed = np.sqrt(squared.sum(axis=0) / valid.sum())
    for hours in [72, 168]:
        blocks, remainder = divmod(len(hourly), hours)
        starts = rng.integers(0, len(hourly) - hours + 1, size=(repetitions, blocks))
        ends = starts + hours
        sample_sums = (sums[ends] - sums[starts]).sum(axis=1)
        sample_counts = (counts[ends] - counts[starts]).sum(axis=1)
        if remainder:
            tail_start = rng.integers(0, len(hourly) - remainder + 1, size=repetitions)
            tail_end = tail_start + remainder
            sample_sums += sums[tail_end] - sums[tail_start]
            sample_counts += counts[tail_end] - counts[tail_start]
        rmse = np.sqrt(sample_sums / sample_counts[:, None])
        for model, reference in comparisons:
            left, right = models.index(model), models.index(reference)
            lower, upper = np.quantile(rmse[:, left] - rmse[:, right], [0.025, 0.975])
            rows.append({
                "model": model, "reference": reference, "block_hours": hours,
                "resamples": repetitions, "n_samples": int(valid.sum()),
                "delta_RMSE": observed[left] - observed[right],
                "lower_95": lower, "upper_95": upper,
            })
    return pd.DataFrame(rows)


def analyze_test(models, predictions, testing, development, output):
    threshold = development[TARGET].quantile(0.9)
    high = predictions["actual"] >= threshold
    importance, pollution, seasonal = [], [], []
    for name, (model, features) in models.items():
        result = permutation_importance(
            model, testing[features], testing[TARGET], random_state=SEED,
            scoring="neg_root_mean_squared_error", n_repeats=5,
        )
        importance.extend(
            {"model": name, "feature": feature, "mean_RMSE_increase": mean, "shuffle_sd": sd}
            for feature, mean, sd in zip(features, result.importances_mean, result.importances_std)
        )
        for label, group in predictions.groupby(high.map({False: "below", True: "high"})):
            errors = group[name] - group["actual"]
            pollution.append({
                "model": name, "group": label, "threshold": threshold,
                **score(group["actual"], group[name]),
                "mean_signed_error": errors.mean(), "fraction_underestimated": (errors < 0).mean(),
            })
        for season, group in predictions.groupby("season"):
            seasonal.append({"model": name, "season": season, **score(group["actual"], group[name])})

    tables = pd.DataFrame(importance), pd.DataFrame(pollution), pd.DataFrame(seasonal)
    for filename, table in zip(
        ["feature_importance.csv", "high_pollution_errors.csv", "seasonal_performance.csv"], tables,
    ):
        table.to_csv(output / filename, index=False)
    paired_bootstrap(predictions, list(models)).to_csv(output / "paired_block_bootstrap.csv", index=False)
    return tables


def save_plot(fig, output, name):
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_figures(metrics, predictions, ablation, importance, pollution, seasonal, output):
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="colorblind")
    names = list(predictions.drop(columns=["actual", "season"]))
    test = metrics[metrics["split"] == "test"].set_index("model").loc[names]
    validation = metrics[metrics["split"] == "validation"].set_index("model").loc[names]
    short = [MODEL_LABELS.get(name, name) for name in names]
    x = np.arange(len(names))

    fig, axis = plt.subplots(figsize=(8, 3.6))
    axis.bar(x - 0.18, validation["RMSE"], width=0.36, label="2013 validation")
    axis.bar(x + 0.18, test["RMSE"], width=0.36, label="2014 test")
    axis.set(xticks=x, xticklabels=short, ylabel=f"RMSE ({UNIT})", ylim=(0, None))
    axis.legend()
    save_plot(fig, output, "model_comparison")

    columns = min(4, len(names))
    rows = (len(names) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(2.5 * columns, 2.65 * rows),
                             sharex=True, sharey=True, squeeze=False)
    actual = predictions["actual"]
    lower = min(0, predictions[names].min().min())
    upper = max(actual.max(), predictions[names].max().max())
    for axis, name, label in zip(axes.flat, names, short):
        axis.scatter(actual, predictions[name], s=4, alpha=0.24, edgecolors="none")
        axis.plot([lower, upper], [lower, upper], "--", color="black", linewidth=0.8)
        axis.set(title=label, xlim=(lower, upper), ylim=(lower, upper))
        axis.tick_params(labelbottom=True)
    for axis in axes.flat[len(names):]:
        axis.set_visible(False)
    fig.supxlabel(f"Observed PM2.5 ({UNIT})")
    fig.supylabel(f"Estimated PM2.5 ({UNIT})")
    save_plot(fig, output, "actual_vs_predicted")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharex=True)
    for axis, name in zip(axes, GRID_FILES):
        top = importance[importance["model"] == name].nlargest(5, "mean_RMSE_increase").iloc[::-1]
        axis.barh(top["feature"], top["mean_RMSE_increase"], xerr=top["shuffle_sd"], capsize=2)
        axis.set_title(MODEL_LABELS[name])
    fig.supxlabel(f"RMSE increase after shuffling ({UNIT})")
    save_plot(fig, output, "feature_importance")

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5), sharey=True)
    for axis, name in zip(axes, GRID_FILES):
        group = ablation[(ablation["model"] == name) & (ablation["split"] == "test")]
        for settings, rows in group.groupby("settings"):
            rows = rows.sort_values("n_features")
            axis.plot(rows["n_features"], rows["RMSE"], "o-", label=settings.capitalize())
        axis.set(title=MODEL_LABELS[name], xticks=[10, 23], xlabel="Input features")
    axes[0].set_ylabel(f"2014 RMSE ({UNIT})")
    axes[-1].legend()
    save_plot(fig, output, "feature_ablation")

    fig, axis = plt.subplots(figsize=(8.5, 3.9))
    for name, label in zip(names, short):
        rows = seasonal[seasonal["model"] == name].set_index("season").loc[SEASONS]
        axis.plot(SEASONS, rows["RMSE"], "o-", label=label)
    axis.set_ylabel(f"2014 RMSE ({UNIT})")
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    save_plot(fig, output, "seasonal_performance")

    fig, axis = plt.subplots(figsize=(8, 3.7))
    for offset, label, text in [(-0.18, "below", "Below cutoff"), (0.18, "high", "At or above cutoff")]:
        rows = pollution[pollution["group"] == label].set_index("model").loc[names]
        axis.bar(x + offset, rows["mean_signed_error"], 0.36, label=text)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(xticks=x, xticklabels=short, ylabel=f"Mean signed error ({UNIT})")
    axis.legend()
    save_plot(fig, output, "high_pollution_errors")
