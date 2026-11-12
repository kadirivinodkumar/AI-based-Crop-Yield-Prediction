"""
Feature Engineering Pipeline
==============================
Computes NDVI/EVI from multispectral patches, normalises images and weather,
and builds flat ML feature vectors for Random Forest / SVR baselines.
"""

import os
import sys
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR, NUM_WEATHER_FEATURES


# ── Vegetation indices ────────────────────────────────────────────────────────
def compute_vegetation_indices(image_sequence):
    """
    Compute NDVI and EVI from image sequence (N, T, H, W, C).
    Channel layout: [Red=0, NIR=1, Blue=2, Green=3, SWIR=4, NDVI=5, EVI=6]
    """
    red  = image_sequence[..., 0]
    nir  = image_sequence[..., 1]
    blue = image_sequence[..., 2]
    ndvi = (nir - red) / (nir + red + 1e-8)
    evi  = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + 1e-8)
    return ndvi.astype(np.float32), evi.astype(np.float32)


# ── Temporal NDVI statistics ──────────────────────────────────────────────────
def extract_temporal_stats(ndvi_sequence):
    """
    Extract 5 temporal statistics from NDVI sequence.
    ndvi_sequence: (N, T, H, W) → returns (N, 5).
    Features: peak_ndvi, mean_ndvi, ndvi_slope, green_up_week, senescence_week
    """
    spatial_mean = ndvi_sequence.mean(axis=(2, 3))   # (N, T)
    peak_ndvi    = spatial_mean.max(axis=1)
    mean_ndvi    = spatial_mean.mean(axis=1)

    T      = spatial_mean.shape[1]
    t_axis = np.arange(T, dtype=np.float32)
    slope  = np.array([
        np.polyfit(t_axis, spatial_mean[i], 1)[0]
        for i in range(len(spatial_mean))
    ], dtype=np.float32)

    green_up = spatial_mean.argmax(axis=1).astype(np.float32)

    half_peak = peak_ndvi / 2.0
    senescence = np.array([
        next((t for t in range(int(green_up[i]), T)
              if spatial_mean[i, t] < half_peak[i]), T - 1)
        for i in range(len(spatial_mean))
    ], dtype=np.float32)

    return np.stack([peak_ndvi, mean_ndvi, slope, green_up, senescence], axis=1)


# ── Scalers ───────────────────────────────────────────────────────────────────
def normalize_weather(weather_data, fit=True, scaler=None):
    """StandardScaler for weather time-series (N, T, F)."""
    N, T, F = weather_data.shape
    flat = weather_data.reshape(-1, F)
    if fit:
        scaler = StandardScaler()
        flat_s = scaler.fit_transform(flat)
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(scaler, os.path.join(MODEL_DIR, "weather_scaler.pkl"))
    else:
        flat_s = scaler.transform(flat)
    return flat_s.reshape(N, T, F).astype(np.float32), scaler


def normalize_images(image_data, fit=True, scaler=None):
    """MinMaxScaler per channel for image data (N, T, H, W, C)."""
    N, T, H, W, C = image_data.shape
    flat = image_data.reshape(-1, C)
    if fit:
        scaler = MinMaxScaler()
        flat_s = scaler.fit_transform(flat)
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(scaler, os.path.join(MODEL_DIR, "image_scaler.pkl"))
    else:
        flat_s = scaler.transform(flat)
    return flat_s.reshape(N, T, H, W, C).astype(np.float32), scaler


# ── ML flat features ──────────────────────────────────────────────────────────
def create_ml_features(ndvi_stats, weather_norm):
    """
    Flat feature vector for RF / SVR baselines.
    ndvi_stats: (N, 5), weather_norm: (N, T, F) → (N, 5 + T*F)
    """
    N = len(ndvi_stats)
    return np.concatenate(
        [ndvi_stats, weather_norm.reshape(N, -1)], axis=1
    ).astype(np.float32)


# ── Full pipeline ─────────────────────────────────────────────────────────────
def prepare_features(images, weather, fit_scalers=True):
    """Run full feature preparation and return all arrays + scalers."""
    ndvi, evi      = compute_vegetation_indices(images)
    ndvi_stats     = extract_temporal_stats(ndvi)
    images_norm, img_sc = normalize_images(images,  fit=fit_scalers)
    weather_norm, wx_sc = normalize_weather(weather, fit=fit_scalers)
    ml_features    = create_ml_features(ndvi_stats, weather_norm)
    return images_norm, weather_norm, ml_features, ndvi_stats, img_sc, wx_sc
