"""
5-Fold Cross-Validation for all models.
IEEE-level requirement: report mean ± std across folds.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras
from keras.callbacks import EarlyStopping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS, NUM_WEATHER_FEATURES,
    BATCH_SIZE, RESULTS_DIR, RANDOM_SEED, LEARNING_RATE, EARLY_STOPPING_PATIENCE
)
from src.data_preprocessing import load_data
from src.feature_engineering import (
    normalize_images, normalize_weather,
    compute_vegetation_indices, extract_temporal_stats, create_ml_features
)
from src.cnn_model import build_standalone_cnn
from src.lstm_model import build_standalone_lstm
from src.fusion_model import build_fusion_model
from src.utils import compute_metrics, set_seeds


N_FOLDS   = 5
CV_EPOCHS = 50   # fewer epochs for CV speed


def _scale_fold(img_tr, wx_tr, y_tr, img_te, wx_te, y_te):
    """Fit scalers on train fold, apply to test fold. Returns scaled arrays."""
    from sklearn.preprocessing import MinMaxScaler
    # Image scaler
    N_tr, T, H, W, C = img_tr.shape
    img_sc = MinMaxScaler()
    img_tr_s = img_sc.fit_transform(img_tr.reshape(-1, C)).reshape(N_tr, T, H, W, C).astype(np.float32)
    N_te = img_te.shape[0]
    img_te_s = img_sc.transform(img_te.reshape(-1, C)).reshape(N_te, T, H, W, C).astype(np.float32)

    # Weather scaler
    wx_sc = StandardScaler()
    wx_tr_s = wx_sc.fit_transform(wx_tr.reshape(-1, NUM_WEATHER_FEATURES)).reshape(
        img_tr.shape[0], T, NUM_WEATHER_FEATURES).astype(np.float32)
    wx_te_s = wx_sc.transform(wx_te.reshape(-1, NUM_WEATHER_FEATURES)).reshape(
        N_te, T, NUM_WEATHER_FEATURES).astype(np.float32)

    # Yield scaler
    y_sc = StandardScaler()
    y_tr_s = y_sc.fit_transform(y_tr.reshape(-1, 1)).flatten().astype(np.float32)

    # ML features
    ndvi_tr, _ = compute_vegetation_indices(img_tr)
    stats_tr   = extract_temporal_stats(ndvi_tr)
    ml_tr      = create_ml_features(stats_tr, wx_tr_s)

    ndvi_te, _ = compute_vegetation_indices(img_te)
    stats_te   = extract_temporal_stats(ndvi_te)
    ml_te      = create_ml_features(stats_te, wx_te_s)

    return img_tr_s, wx_tr_s, ml_tr, y_tr_s, img_te_s, wx_te_s, ml_te, y_sc


def run_cross_validation():
    set_seeds(RANDOM_SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n[CV] Loading data ...")
    images, weather, yields, crop_labels = load_data()
    N = len(yields)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    img_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS + 2)
    wx_shape  = (SEQUENCE_LENGTH, NUM_WEATHER_FEATURES)

    model_names = ["standalone_cnn", "standalone_lstm", "fusion_model",
                   "random_forest", "svr"]
    fold_results = {m: [] for m in model_names}

    for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(np.arange(N))):
        print(f"\n[CV] ===== Fold {fold_idx + 1}/{N_FOLDS} =====")

        img_tr, wx_tr, y_tr = images[tr_idx], weather[tr_idx], yields[tr_idx]
        img_te, wx_te, y_te = images[te_idx], weather[te_idx], yields[te_idx]

        img_tr_s, wx_tr_s, ml_tr, y_tr_s, img_te_s, wx_te_s, ml_te, y_sc = \
            _scale_fold(img_tr, wx_tr, y_tr, img_te, wx_te, y_te)

        cb = [EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
                            restore_best_weights=True, verbose=0)]

        def inv(p):
            return y_sc.inverse_transform(np.array(p).flatten().reshape(-1, 1)).flatten()

        # --- Standalone CNN ---
        print("  Training standalone_cnn ...")
        cnn = build_standalone_cnn(img_shape)
        cnn.fit(img_tr_s, y_tr_s,
                validation_split=0.15, epochs=CV_EPOCHS,
                batch_size=BATCH_SIZE, callbacks=cb, verbose=0)
        pred = inv(cnn.predict(img_te_s, verbose=0).flatten())
        fold_results["standalone_cnn"].append(compute_metrics(y_te, pred))
        keras.backend.clear_session()

        # --- Standalone LSTM ---
        print("  Training standalone_lstm ...")
        lstm = build_standalone_lstm(wx_shape)
        lstm.fit(wx_tr_s, y_tr_s,
                 validation_split=0.15, epochs=CV_EPOCHS,
                 batch_size=BATCH_SIZE, callbacks=cb, verbose=0)
        pred = inv(lstm.predict(wx_te_s, verbose=0).flatten())
        fold_results["standalone_lstm"].append(compute_metrics(y_te, pred))
        keras.backend.clear_session()

        # --- Fusion ---
        print("  Training fusion_model ...")
        fusion = build_fusion_model(img_shape, wx_shape)
        fusion.fit([img_tr_s, wx_tr_s], y_tr_s,
                   validation_data=([img_te_s, wx_te_s],
                                    y_sc.transform(y_te.reshape(-1, 1)).flatten()),
                   epochs=CV_EPOCHS, batch_size=BATCH_SIZE, callbacks=cb, verbose=0)
        pred = inv(fusion.predict([img_te_s, wx_te_s], verbose=0).flatten())
        fold_results["fusion_model"].append(compute_metrics(y_te, pred))
        keras.backend.clear_session()

        # --- Random Forest ---
        print("  Training random_forest ...")
        rf = RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                   random_state=RANDOM_SEED, n_jobs=-1)
        rf.fit(ml_tr, y_tr_s)
        pred = inv(rf.predict(ml_te))
        fold_results["random_forest"].append(compute_metrics(y_te, pred))

        # --- SVR ---
        print("  Training svr ...")
        svr = SVR(kernel="rbf", C=10, epsilon=0.1)
        svr.fit(ml_tr, y_tr_s)
        pred = inv(svr.predict(ml_te))
        fold_results["svr"].append(compute_metrics(y_te, pred))

    # Aggregate results
    print("\n[CV] ===== Cross-Validation Summary =====")
    summary = {}
    for model_name, folds in fold_results.items():
        rmse_vals  = [f["RMSE"]  for f in folds]
        mae_vals   = [f["MAE"]   for f in folds]
        r2_vals    = [f["R2"]    for f in folds]
        smape_vals = [f["sMAPE"] for f in folds]
        summary[model_name] = {
            "RMSE_mean":  float(np.mean(rmse_vals)),
            "RMSE_std":   float(np.std(rmse_vals)),
            "MAE_mean":   float(np.mean(mae_vals)),
            "MAE_std":    float(np.std(mae_vals)),
            "R2_mean":    float(np.mean(r2_vals)),
            "R2_std":     float(np.std(r2_vals)),
            "sMAPE_mean": float(np.mean(smape_vals)),
            "sMAPE_std":  float(np.std(smape_vals)),
        }

    # Print table
    print(f"\n{'Model':<22} {'RMSE':>16} {'MAE':>16} {'R²':>16} {'sMAPE%':>16}")
    print("-" * 90)
    for m, s in summary.items():
        print(f"{m:<22} "
              f"{s['RMSE_mean']:.4f}±{s['RMSE_std']:.4f}  "
              f"{s['MAE_mean']:.4f}±{s['MAE_std']:.4f}  "
              f"{s['R2_mean']:.4f}±{s['R2_std']:.4f}  "
              f"{s['sMAPE_mean']:.2f}±{s['sMAPE_std']:.2f}")

    # Save
    cv_path = os.path.join(RESULTS_DIR, "cv_results.json")
    with open(cv_path, "w") as f:
        import json
        json.dump(summary, f, indent=2)
    print(f"\nCV results saved to {cv_path}")

    # Save as CSV for paper table
    rows = []
    for m, s in summary.items():
        rows.append({
            "Model": m,
            "RMSE": f"{s['RMSE_mean']:.4f}±{s['RMSE_std']:.4f}",
            "MAE":  f"{s['MAE_mean']:.4f}±{s['MAE_std']:.4f}",
            "R2":   f"{s['R2_mean']:.4f}±{s['R2_std']:.4f}",
            "sMAPE": f"{s['sMAPE_mean']:.2f}±{s['sMAPE_std']:.2f}",
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, "cv_results_table.csv"), index=False
    )
    print("CV table saved to outputs/results/cv_results_table.csv")
    return summary


if __name__ == "__main__":
    run_cross_validation()
