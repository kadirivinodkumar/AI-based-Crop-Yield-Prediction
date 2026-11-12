"""
Data Preprocessing Pipeline
============================
Abstract alignment:
  - Satellite imagery: Real Sentinel-2 multispectral patches (5 bands + NDVI + EVI = 7 channels)
    downloaded via Microsoft Planetary Computer.
  - Weather time-series: rainfall, temperature, humidity, soil_moisture (4 features)
  - Vegetation indices: NDVI and EVI extracted from real Sentinel-2 bands
  - Dataset: FAO/Kaggle yield_df.csv — 6 crops, 200 records each = 1200 total
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROCESSED_DIR, RAW_DIR, SATELLITE_DIR,
    NUM_BANDS, NUM_WEATHER_FEATURES,
    IMAGE_HEIGHT, IMAGE_WIDTH,
    SEQUENCE_LENGTH, RANDOM_SEED,
    TEST_SPLIT, VALIDATION_SPLIT,
)

CROP_MAP = {
    "Maize":       {"label": "Maize",   "class": 0},
    "Rice, paddy": {"label": "Rice",    "class": 1},
    "Wheat":       {"label": "Wheat",   "class": 2},
    "Soybeans":    {"label": "Soybean", "class": 3},
    "Sorghum":     {"label": "Sorghum", "class": 4},
    "Potatoes":    {"label": "Potato",  "class": 5},
}

SEASON_CURVE = [0.08, 0.18, 0.35, 0.52, 0.68, 0.80,
                0.84, 0.76, 0.60, 0.42, 0.25, 0.10]


# Physical valid ranges per channel: [Red, NIR, Blue, Green, SWIR, NDVI, EVI]
CHANNEL_CLIP = [
    (0.0, 1.0),   # Red   (reflectance 0-1)
    (0.0, 1.0),   # NIR
    (0.0, 1.0),   # Blue
    (0.0, 1.0),   # Green
    (0.0, 1.0),   # SWIR
    (-1.0, 1.0),  # NDVI
    (-1.0, 1.0),  # EVI  — clip outliers from division instability
]


def _load_satellite_images():
    """Load, clip to physical ranges, and normalize real Sentinel-2 images."""
    sat_path = os.path.join(SATELLITE_DIR, "satellite_images.npy")
    if not os.path.exists(sat_path):
        raise FileNotFoundError(
            f"Real satellite images not found at {sat_path}\n"
            "Run src/sentinel2_downloader.py first."
        )
    print(f"  Loading real Sentinel-2 images from {sat_path} ...")
    images = np.load(sat_path).astype(np.float32)   # (1200, 12, 32, 32, 7)

    # Clip each channel to physically valid range before normalizing
    for c, (lo, hi) in enumerate(CHANNEL_CLIP):
        images[..., c] = np.clip(images[..., c], lo, hi)

    # Normalize each channel to [0, 1]
    for c in range(images.shape[-1]):
        cmin = images[..., c].min()
        cmax = images[..., c].max()
        images[..., c] = (images[..., c] - cmin) / (cmax - cmin + 1e-8)

    print(f"  Satellite images shape: {images.shape} | clipped + normalized to [0,1]")
    return images


def generate_and_save():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    # ── 1. Load real Sentinel-2 images ───────────────────────────────────────
    print("[1/5] Loading real Sentinel-2 satellite images ...")
    sat_images = _load_satellite_images()   # (1200, 12, 32, 32, 7)
    # sat_images is ordered: 200 Maize, 200 Rice, 200 Wheat, 200 Soybean, 200 Sorghum, 200 Potato

    # ── 2. Load FAO/Kaggle yield records ─────────────────────────────────────
    print("[2/5] Loading yield_df.csv ...")
    df = pd.read_csv(os.path.join(RAW_DIR, "yield_df.csv"))
    df = df[df["Item"].isin(CROP_MAP.keys())].dropna().reset_index(drop=True)
    df["yield_t_ha"] = df["hg/ha_yield"] / 10_000.0

    # Per-crop outlier removal (5th-95th percentile) + hard cap 12 t/ha
    rows = []
    for crop, grp in df.groupby("Item"):
        lo = grp["yield_t_ha"].quantile(0.05)
        hi = min(grp["yield_t_ha"].quantile(0.95), 12.0)
        rows.append(grp[(grp["yield_t_ha"] >= lo) & (grp["yield_t_ha"] <= hi)])
    df = pd.concat(rows).reset_index(drop=True)

    df["crop_class"] = df["Item"].map(lambda x: CROP_MAP[x]["class"])
    df["crop_label"] = df["Item"].map(lambda x: CROP_MAP[x]["label"])

    # Stratified sample: exactly 200 per crop = 1200 total
    df = (df.groupby("Item", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), 200), random_state=RANDOM_SEED))
            .reset_index(drop=True))
    # Sort by crop_class so indices align with satellite images order
    df = df.sort_values("crop_class").reset_index(drop=True)

    N = len(df)
    print(f"  {N} records | yield range: "
          f"{df['yield_t_ha'].min():.2f}-{df['yield_t_ha'].max():.2f} t/ha")
    print(f"  Crop counts:\n{df['Item'].value_counts().to_string()}")

    # ── 3. Build final arrays ─────────────────────────────────────────────────
    print(f"[3/5] Fusing satellite images with weather sequences ...")
    C_img = NUM_BANDS + 2   # 7 channels
    # NUM_WEATHER_FEATURES = 6: rainfall, temp, humidity, soil_moisture, pesticides_norm, year_norm
    images      = np.zeros((N, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, C_img), dtype=np.float32)
    weather     = np.zeros((N, SEQUENCE_LENGTH, NUM_WEATHER_FEATURES),             dtype=np.float32)
    yields      = np.zeros(N,  dtype=np.float32)
    crop_labels = np.zeros(N,  dtype=np.int32)
    ndvi_store  = np.zeros((N, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH),        dtype=np.float32)

    # Normalise pesticides and year globally for the dataset
    pest_vals = df["pesticides_tonnes"].values.astype(np.float32)
    year_vals = df["Year"].values.astype(np.float32)
    pest_norm = (pest_vals - pest_vals.min()) / (pest_vals.max() - pest_vals.min() + 1e-8)
    year_norm = (year_vals - year_vals.min()) / (year_vals.max() - year_vals.min() + 1e-8)

    for idx, (i, row) in enumerate(df.iterrows()):
        rainfall  = float(row["average_rain_fall_mm_per_year"])
        temp      = float(row["avg_temp"])
        yield_val = float(row["yield_t_ha"])
        crop_cls  = int(row["crop_class"])
        pest_n    = float(pest_norm[idx])
        year_n    = float(year_norm[idx])

        crop_labels[idx] = crop_cls
        yields[idx]      = yield_val

        # ── Real Sentinel-2 image sequence ───────────────────────────────────
        images[idx]     = sat_images[idx]              # (12, 32, 32, 7)
        ndvi_store[idx] = sat_images[idx, :, :, :, 5]  # channel 5 = NDVI

        # ── Weather sequence (6 features) ────────────────────────────────────
        for t in range(SEQUENCE_LENGTH):
            s       = SEASON_CURVE[t]
            rain_w  = float(np.clip(rainfall / 12.0 * s + rng.normal(0, 7.0), 0, rainfall / 4))
            temp_w  = float(temp + 4.0 * s + rng.normal(0, 1.2))
            humid_w = float(np.clip(45.0 + (rainfall / 1500.0) * 40.0 + s * 10.0
                                    + rng.normal(0, 3.5), 20.0, 98.0))
            soil_w  = float(np.clip(0.15 + (rainfall / 1500.0) * 0.35 + s * 0.15
                                    + rng.normal(0, 0.02), 0.05, 0.65))
            weather[idx, t] = [rain_w, temp_w, humid_w, soil_w, pest_n, year_n]

        if (idx + 1) % 400 == 0:
            print(f"  {idx+1}/{N} processed ...")

    # ── 4. Save ───────────────────────────────────────────────────────────────
    print("[4/5] Saving processed arrays ...")
    from config import NUM_CROPS
    crop_onehot = np.eye(NUM_CROPS, dtype=np.float32)[crop_labels]  # (N, 6)
    np.save(os.path.join(PROCESSED_DIR, "images.npy"),      images)
    np.save(os.path.join(PROCESSED_DIR, "weather.npy"),     weather)
    np.save(os.path.join(PROCESSED_DIR, "yields.npy"),      yields)
    np.save(os.path.join(PROCESSED_DIR, "crop_labels.npy"), crop_labels)
    np.save(os.path.join(PROCESSED_DIR, "crop_onehot.npy"), crop_onehot)
    np.save(os.path.join(PROCESSED_DIR, "ndvi.npy"),        ndvi_store)

    df[[
        "Area", "Item", "Year",
        "average_rain_fall_mm_per_year", "avg_temp",
        "pesticides_tonnes", "yield_t_ha", "crop_class", "crop_label",
    ]].to_csv(os.path.join(PROCESSED_DIR, "field_metadata.csv"), index=False)

    print(f"[5/5] Done.")
    print(f"  images  : {images.shape}  (real Sentinel-2)")
    print(f"  weather : {weather.shape}")
    print(f"  yields  : {yields.shape}  mean={yields.mean():.3f}  std={yields.std():.3f}")
    return images, weather, yields, crop_labels


def load_data():
    images      = np.load(os.path.join(PROCESSED_DIR, "images.npy"))
    weather     = np.load(os.path.join(PROCESSED_DIR, "weather.npy"))
    yields      = np.load(os.path.join(PROCESSED_DIR, "yields.npy"))
    crop_labels = np.load(os.path.join(PROCESSED_DIR, "crop_labels.npy"))
    crop_onehot = np.load(os.path.join(PROCESSED_DIR, "crop_onehot.npy"))
    return images, weather, yields, crop_labels, crop_onehot


def split_data(images, weather, yields, crop_labels, crop_onehot):
    rng = np.random.default_rng(RANDOM_SEED)
    N   = len(yields)
    idx = rng.permutation(N)

    n_test = int(N * TEST_SPLIT)
    n_val  = int(N * VALIDATION_SPLIT)

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    splits = {}
    for name, i in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        splits[name] = {
            "images":      images[i],
            "weather":     weather[i],
            "yields":      yields[i],
            "crop_labels": crop_labels[i],
            "crop_onehot": crop_onehot[i],
        }
    print(f"Split  train:{len(train_idx)}  val:{len(val_idx)}  test:{len(test_idx)}")
    return splits


if __name__ == "__main__":
    generate_and_save()
