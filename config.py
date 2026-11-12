"""
Configuration — matches abstract exactly:
- Weather: rainfall, temperature, humidity, soil moisture (4 features)
- Images: Sentinel-2 style multispectral (5 bands + NDVI + EVI = 7 channels)
- Sequence: 12 weeks (one growing season)
"""
import os

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
RAW_DIR       = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SAMPLE_DIR      = os.path.join(DATA_DIR, "sample")
SATELLITE_DIR   = os.path.join(DATA_DIR, "satellite_images")
OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs")
MODEL_DIR     = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR      = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR   = os.path.join(OUTPUT_DIR, "results")

# ── Data ──────────────────────────────────────────────────────────────────────
NUM_FIELDS           = 1200          # 200 per crop × 6 crops
SEQUENCE_LENGTH      = 12            # 12 weekly timesteps per growing season
IMAGE_HEIGHT         = 32            # 32×32 spatial patch (EuroSAT-style)
IMAGE_WIDTH          = 32
NUM_BANDS            = 5             # Red, NIR, Blue, Green, SWIR
# Total image channels = NUM_BANDS + 2 (NDVI + EVI) = 7

# Weather features — now 6: rainfall, temperature, humidity, soil_moisture, pesticides_norm, year_norm

# ── Model ─────────────────────────────────────────────────────────────────────
CNN_FILTERS              = [64, 128, 256]   # deeper ResNet-style encoder
LSTM_UNITS               = [128, 64]         # wider LSTM
DROPOUT_RATE             = 0.25
DENSE_UNITS              = [256, 128]
LEARNING_RATE            = 5e-4
BATCH_SIZE               = 16               # smaller batch → better gradient signal on 1200 samples
EPOCHS                   = 100
EARLY_STOPPING_PATIENCE  = 20

# ── Enhanced features ─────────────────────────────────────────────────────────
NUM_WEATHER_FEATURES     = 6               # +pesticides_norm +year_norm added
NUM_CROPS                = 6               # one-hot crop embedding dimension

# ── Splits ────────────────────────────────────────────────────────────────────
VALIDATION_SPLIT = 0.15
TEST_SPLIT       = 0.15

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
