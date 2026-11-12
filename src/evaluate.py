"""
IEEE-level evaluation pipeline.
Generates: comparison table, predicted-vs-actual, per-crop breakdown,
ablation study, bootstrap CI, residual plot, training curves, NDVI trend.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR, PLOT_DIR, RESULTS_DIR, RANDOM_SEED, SEQUENCE_LENGTH
from src.data_preprocessing import load_data, split_data, CROP_MAP
from src.feature_engineering import (
    normalize_images, normalize_weather, compute_vegetation_indices,
    extract_temporal_stats, create_ml_features
)
from src.utils import (
    compute_metrics, format_results_table, create_output_dirs,
    bootstrap_ci, compute_per_crop_metrics
)
from src.fusion_model import SliceChannel  # noqa: F401 — must be imported to register custom layer

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11})

LABEL_NAMES = {v["class"]: v["label"] for v in CROP_MAP.values()}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_all_models():
    models = {}
    for name in ["standalone_cnn", "standalone_lstm", "fusion_model"]:
        path = os.path.join(MODEL_DIR, f"{name}.keras")
        if os.path.exists(path):
            models[name] = keras.models.load_model(path, safe_mode=False)
            print(f"Loaded {name}")
    for name in ["random_forest", "svr"]:
        path = os.path.join(MODEL_DIR, f"{name}.pkl")
        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"Loaded {name}")
    return models


def get_test_data():
    images, weather, yields, crop_labels, crop_onehot = load_data()
    splits = split_data(images, weather, yields, crop_labels, crop_onehot)

    img_sc   = joblib.load(os.path.join(MODEL_DIR, "image_scaler.pkl"))
    wx_sc    = joblib.load(os.path.join(MODEL_DIR, "weather_scaler.pkl"))
    yield_sc = joblib.load(os.path.join(MODEL_DIR, "yield_scaler.pkl"))

    img_te, _ = normalize_images(splits["test"]["images"],  fit=False, scaler=img_sc)
    wx_te, _  = normalize_weather(splits["test"]["weather"], fit=False, scaler=wx_sc)

    ndvi, _   = compute_vegetation_indices(splits["test"]["images"])
    stats     = extract_temporal_stats(ndvi)
    ml_te     = create_ml_features(stats, wx_te)

    y_te_raw  = splits["test"]["yields"]
    cl_te     = splits["test"]["crop_labels"]
    crop_te   = splits["test"]["crop_onehot"]

    # NDVI seasonal plot uses full dataset
    img_all, _        = normalize_images(images, fit=False, scaler=img_sc)
    ndvi_all, _       = compute_vegetation_indices(img_all)
    ndvi_spatial_mean = ndvi_all.mean(axis=(2, 3))

    return img_te, wx_te, ml_te, y_te_raw, ndvi_spatial_mean, cl_te, yield_sc, crop_te


def predict_all(models, img_te, wx_te, ml_te, yield_sc, crop_te):
    preds = {}

    def inv(p):
        return yield_sc.inverse_transform(
            np.array(p).flatten().reshape(-1, 1)
        ).flatten()

    if "standalone_cnn" in models:
        preds["standalone_cnn"] = inv(
            models["standalone_cnn"].predict(img_te, verbose=0).flatten()
        )
    if "standalone_lstm" in models:
        preds["standalone_lstm"] = inv(
            models["standalone_lstm"].predict(wx_te, verbose=0).flatten()
        )
    if "fusion_model" in models:
        preds["fusion_model"] = inv(
            models["fusion_model"].predict([img_te, wx_te, crop_te], verbose=0).flatten()
        )
    if "random_forest" in models:
        preds["random_forest"] = inv(models["random_forest"].predict(ml_te))
    if "svr" in models:
        preds["svr"] = inv(models["svr"].predict(ml_te))

    return preds


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_comparison_table(results, save_path):
    df = pd.DataFrame(results).T.reset_index()
    df.columns = ["Model", "RMSE", "MAE", "R²", "sMAPE%"]
    df = df.round(4)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    plt.title("Model Performance Comparison (Test Set)", fontsize=13, pad=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_predicted_vs_actual(y_true, y_pred, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color="steelblue", label="Predictions")
    lims = [min(y_true.min(), y_pred.min()) - 0.2,
            max(y_true.max(), y_pred.max()) + 0.2]
    ax.plot(lims, lims, "r--", label="Perfect fit")
    ax.set_xlabel("Actual Yield (t/ha)")
    ax.set_ylabel("Predicted Yield (t/ha)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_residuals(y_true, y_pred, save_path):
    residuals = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.4, s=18, color="steelblue")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted Yield (t/ha)")
    axes[0].set_ylabel("Residual (t/ha)")
    axes[0].set_title("Residual vs Predicted")
    axes[1].hist(residuals, bins=30, color="steelblue", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Residual (t/ha)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")
    plt.suptitle("Fusion Model — Residual Analysis", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_per_crop_metrics(y_true, preds_dict, crop_labels, save_path):
    """Bar chart of R² per crop for each model."""
    model_names = list(preds_dict.keys())
    crop_names  = sorted(set(LABEL_NAMES.values()))
    r2_matrix   = np.zeros((len(model_names), len(crop_names)))

    for mi, mname in enumerate(model_names):
        per_crop = compute_per_crop_metrics(
            y_true, preds_dict[mname], crop_labels, LABEL_NAMES
        )
        for ci, cname in enumerate(crop_names):
            r2_matrix[mi, ci] = per_crop.get(cname, {}).get("R2", 0.0)

    x = np.arange(len(crop_names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["steelblue", "darkorange", "green", "red", "purple"]
    for mi, (mname, color) in enumerate(zip(model_names, colors)):
        ax.bar(x + mi * width, r2_matrix[mi], width, label=mname, color=color, alpha=0.8)
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(crop_names, rotation=20, ha="right")
    ax.set_ylabel("R²")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Crop R² by Model")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_ablation_study(y_true, preds_dict, save_path):
    """
    Ablation: compare CNN-only, LSTM-only, Fusion.
    Shows contribution of each branch.
    """
    ablation_models = ["standalone_cnn", "standalone_lstm", "fusion_model"]
    metrics_list = []
    for m in ablation_models:
        if m in preds_dict:
            met = compute_metrics(y_true, preds_dict[m])
            metrics_list.append({"Model": m, **met})

    if not metrics_list:
        return

    df = pd.DataFrame(metrics_list).set_index("Model")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ["steelblue", "darkorange", "green"]
    for ax, metric in zip(axes, ["RMSE", "R2", "sMAPE"]):
        vals = [df.loc[m, metric] if m in df.index else 0 for m in ablation_models]
        bars = ax.bar(ablation_models, vals, color=colors, alpha=0.85)
        ax.set_title(metric)
        ax.set_xticklabels(["CNN", "LSTM", "Fusion"], rotation=15)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.suptitle("Ablation Study — CNN vs LSTM vs Fusion", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_bootstrap_ci(y_true, y_pred, model_name, save_path):
    """Plot bootstrap confidence intervals for all metrics."""
    ci = bootstrap_ci(y_true, y_pred, n_boot=500, seed=RANDOM_SEED)
    metrics = list(ci.keys())
    means   = [ci[m][0] for m in metrics]
    lowers  = [ci[m][0] - ci[m][1] for m in metrics]
    uppers  = [ci[m][2] - ci[m][0] for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(metrics))
    ax.bar(x, means, color="steelblue", alpha=0.8, label="Mean")
    ax.errorbar(x, means, yerr=[lowers, uppers], fmt="none",
                color="black", capsize=6, linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title(f"Bootstrap 95% CI — {model_name}")
    ax.set_ylabel("Metric Value")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_training_curves(histories, save_path):
    deep_models = ["standalone_cnn", "standalone_lstm", "fusion_model"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, name in zip(axes, deep_models):
        if name not in histories:
            ax.set_visible(False)
            continue
        h = histories[name]
        ax.plot(h["loss"], label="Train Loss")
        ax.plot(h["val_loss"], label="Val Loss")
        ax.set_title(name.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=8)
    plt.suptitle("Training Loss Curves", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_ndvi_seasonal(ndvi_spatial_mean, save_path):
    mean_curve = ndvi_spatial_mean.mean(axis=0)
    std_curve  = ndvi_spatial_mean.std(axis=0)
    weeks = np.arange(1, len(mean_curve) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(weeks, mean_curve, color="green", linewidth=2, label="Mean NDVI")
    ax.fill_between(weeks, mean_curve - std_curve, mean_curve + std_curve,
                    alpha=0.3, color="green", label="±1 std")
    ax.set_xlabel("Week of Growing Season")
    ax.set_ylabel("NDVI")
    ax.set_title("Seasonal NDVI Trend (All Fields)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_feature_importance(models, save_path):
    if "random_forest" not in models:
        return
    rf = models["random_forest"]
    importances = rf.feature_importances_
    top_n = min(20, len(importances))
    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(top_n), importances[idx], color="olivedrab")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([f"F{i}" for i in idx], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Importance")
    ax.set_title("Random Forest — Top Feature Importances")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_yield_distribution(y_true, y_pred, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_true, bins=30, alpha=0.6, color="steelblue", label="Actual")
    ax.hist(y_pred, bins=30, alpha=0.6, color="darkorange", label="Predicted")
    ax.set_xlabel("Yield (t/ha)")
    ax.set_ylabel("Count")
    ax.set_title("Yield Distribution — Actual vs Predicted (Fusion)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    create_output_dirs()

    print("\n[Step 1] Loading models ...")
    models = load_all_models()

    print("\n[Step 2] Preparing test data ...")
    img_te, wx_te, ml_te, y_te, ndvi_spatial_mean, cl_te, yield_sc, crop_te = get_test_data()
    print(f"  y_te range: min={y_te.min():.2f}  max={y_te.max():.2f}  mean={y_te.mean():.2f}")

    print("\n[Step 3] Generating predictions ...")
    preds = predict_all(models, img_te, wx_te, ml_te, yield_sc, crop_te)
    for name, p in preds.items():
        print(f"  {name}: pred min={p.min():.2f}  max={p.max():.2f}  mean={p.mean():.2f}")

    print("\n[Step 4] Computing metrics (sMAPE) ...")
    results = {name: compute_metrics(y_te, p) for name, p in preds.items()}
    format_results_table(results)

    df = pd.DataFrame(results).T
    df.index.name = "Model"
    csv_path = os.path.join(RESULTS_DIR, "all_model_results.csv")
    df.to_csv(csv_path)
    print(f"Results CSV saved to {csv_path}")

    print("\n[Step 5] Generating IEEE-level plots ...")

    # 1. Comparison table
    plot_comparison_table(results, os.path.join(PLOT_DIR, "comparison_table.png"))

    # 2. Predicted vs Actual (all models)
    for mname, pred in preds.items():
        plot_predicted_vs_actual(
            y_te, pred,
            f"{mname.replace('_', ' ').title()} — Predicted vs Actual",
            os.path.join(PLOT_DIR, f"pred_vs_actual_{mname}.png")
        )

    # 3. Residual analysis (fusion)
    if "fusion_model" in preds:
        plot_residuals(y_te, preds["fusion_model"],
                       os.path.join(PLOT_DIR, "residuals_fusion.png"))

    # 4. Per-crop R² breakdown
    plot_per_crop_metrics(y_te, preds, cl_te,
                          os.path.join(PLOT_DIR, "per_crop_r2.png"))

    # 5. Ablation study
    plot_ablation_study(y_te, preds,
                        os.path.join(PLOT_DIR, "ablation_study.png"))

    # 6. Bootstrap CI (fusion)
    if "fusion_model" in preds:
        plot_bootstrap_ci(y_te, preds["fusion_model"], "Fusion Model",
                          os.path.join(PLOT_DIR, "bootstrap_ci_fusion.png"))

    # 7. Yield distribution
    if "fusion_model" in preds:
        plot_yield_distribution(y_te, preds["fusion_model"],
                                os.path.join(PLOT_DIR, "yield_distribution.png"))

    # 8. Training curves
    hist_path = os.path.join(RESULTS_DIR, "training_histories.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            histories = json.load(f)
        plot_training_curves(histories, os.path.join(PLOT_DIR, "training_loss_curves.png"))

    # 9. NDVI seasonal trend
    plot_ndvi_seasonal(ndvi_spatial_mean, os.path.join(PLOT_DIR, "ndvi_seasonal_trend.png"))

    # 10. Feature importance
    plot_feature_importance(models, os.path.join(PLOT_DIR, "feature_importance.png"))

    print("\nEvaluation complete. All plots saved to outputs/plots/")


if __name__ == "__main__":
    main()
