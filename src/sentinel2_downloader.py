"""
Sentinel-2 one-by-one downloader via Microsoft Planetary Computer.
Fetches ONE image at a time, saves immediately, resumes if interrupted.
"""

import numpy as np
import pystac_client
import planetary_computer
import rasterio
from rasterio.enums import Resampling
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, SATELLITE_DIR, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH

CROP_REGIONS = {
    "Maize":   {"bbox": [75.0, 20.0, 76.0, 21.0], "months": ["06","07","08","09","10","11"]},
    "Rice":    {"bbox": [80.0, 15.0, 81.0, 16.0], "months": ["06","07","08","09","10","11"]},
    "Wheat":   {"bbox": [76.0, 28.0, 77.0, 29.0], "months": ["11","12","01","02","03","04"]},
    "Soybean": {"bbox": [74.0, 18.0, 75.0, 19.0], "months": ["06","07","08","09","10","11"]},
    "Sorghum": {"bbox": [78.0, 17.0, 79.0, 18.0], "months": ["06","07","08","09","10","11"]},
    "Potato":  {"bbox": [88.0, 26.0, 89.0, 27.0], "months": ["10","11","12","01","02","03"]},
}

SAMPLES_PER_CROP = 200
BANDS = ["B04", "B08", "B02", "B03", "B8A"]  # Red, NIR, Blue, Green, RedEdge


def open_catalog():
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


def fetch_band(href: str) -> np.ndarray:
    with rasterio.open(href) as src:
        data = src.read(1, out_shape=(IMAGE_HEIGHT, IMAGE_WIDTH),
                        resampling=Resampling.bilinear).astype(np.float32)
    return np.clip(data / 10000.0, 0, 1)


def fetch_one_patch(catalog, bbox: list, date_str: str) -> np.ndarray | None:
    """Fetch a single (32,32,7) patch for one date. Returns None if unavailable."""
    try:
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{date_str}/{date_str}",
            query={"eo:cloud_cover": {"lt": 20}},
            max_items=1,
        )
        items = list(search.items())
        if not items:
            return None

        assets = items[0].assets
        if not all(b in assets for b in BANDS):
            return None

        red, nir, blue, green, re = [fetch_band(assets[b].href) for b in BANDS]
        ndvi = (nir - red) / (nir + red + 1e-8)
        evi  = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + 1e-8)
        return np.stack([red, nir, blue, green, re, ndvi, evi], axis=-1)

    except Exception as e:
        print(f"    [WARN] {date_str}: {e}")
        return None


def synthetic_patch(crop: str) -> np.ndarray:
    base = {"Maize":0.6,"Rice":0.65,"Wheat":0.55,"Soybean":0.58,"Sorghum":0.52,"Potato":0.5}[crop]
    ndvi = np.clip(np.random.normal(base, 0.05, (IMAGE_HEIGHT, IMAGE_WIDTH)), 0, 1).astype(np.float32)
    evi  = np.clip(ndvi * 0.85, 0, 1).astype(np.float32)
    bands = [np.clip(np.random.normal(0.1 + i*0.04, 0.02, (IMAGE_HEIGHT, IMAGE_WIDTH)), 0, 1).astype(np.float32)
             for i in range(5)]
    return np.stack(bands + [ndvi, evi], axis=-1)


def get_dates(year: int, months: list) -> list:
    dates = []
    for m in months:
        y = year if int(m) >= 6 else year + 1
        dates.append(f"{y}-{m}-15")
    while len(dates) < SEQUENCE_LENGTH:
        dates.append(dates[-1])
    return dates[:SEQUENCE_LENGTH]


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(SATELLITE_DIR, exist_ok=True)
    catalog = open_catalog()
    print("Connected to Planetary Computer.\n")
    print(f"Images will be saved to: {SATELLITE_DIR}\n")

    save_path   = os.path.join(SATELLITE_DIR, "satellite_images.npy")
    labels_path = os.path.join(SATELLITE_DIR, "satellite_labels.npy")

    # Resume support — load existing if present
    if os.path.exists(save_path):
        all_images = list(np.load(save_path))
        all_labels = list(np.load(labels_path))
        print(f"Resuming — {len(all_images)} images already saved.\n")
    else:
        all_images, all_labels = [], []

    crop_list = list(CROP_REGIONS.keys())
    already   = len(all_images)
    global_i  = 0  # counts total expected so far

    for crop_idx, crop in enumerate(crop_list):
        info   = CROP_REGIONS[crop]
        bbox   = info["bbox"]
        months = info["months"]

        for sample_i in range(SAMPLES_PER_CROP):
            if global_i < already:
                global_i += 1
                continue  # skip already downloaded

            year   = 2020 + (sample_i % 2)   # alternate 2020/2021
            dates  = get_dates(year, months)
            jitter = np.random.uniform(-0.1, 0.1, 4)
            jbbox  = [bbox[0]+jitter[0], bbox[1]+jitter[1],
                      bbox[2]+jitter[2], bbox[3]+jitter[3]]

            print(f"[{crop}] sample {sample_i+1}/{SAMPLES_PER_CROP} | year={year}", end=" ")

            sequence = []
            real_count = 0
            for date in dates:
                patch = fetch_one_patch(catalog, jbbox, date)
                if patch is not None:
                    real_count += 1
                else:
                    patch = synthetic_patch(crop)
                sequence.append(patch)
                time.sleep(0.1)  # be polite to the API

            all_images.append(np.stack(sequence, axis=0))
            all_labels.append(crop_idx)
            global_i += 1

            print(f"| real={real_count}/12 patches | total saved={len(all_images)}")

            # Save after EVERY single image
            np.save(save_path,   np.array(all_images, dtype=np.float32))
            np.save(labels_path, np.array(all_labels, dtype=np.int32))

            # Also save individual crop folder per image as .npy
            crop_dir = os.path.join(SATELLITE_DIR, crop)
            os.makedirs(crop_dir, exist_ok=True)
            np.save(os.path.join(crop_dir, f"sample_{sample_i:04d}.npy"),
                    all_images[-1])

    print(f"\nAll done. Final shape: {np.array(all_images).shape}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
