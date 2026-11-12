import os
import sys
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

import keras
keras.config.enable_unsafe_deserialization()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_DIR, RESULTS_DIR, PROCESSED_DIR,
    SEQUENCE_LENGTH, NUM_WEATHER_FEATURES, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS
)

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

CROPS = {
    "Maize":   {"base": 5.1, "emoji": "🌽"},
    "Rice":    {"base": 4.2, "emoji": "🍚"},
    "Wheat":   {"base": 3.5, "emoji": "🌾"},
    "Soybean": {"base": 2.8, "emoji": "🌱"},
    "Sorghum": {"base": 2.2, "emoji": "🌿"},
    "Potato":  {"base": 4.6, "emoji": "🥔"},
}

CROP_IMAGES = {
    "Maize":   "https://images.unsplash.com/photo-1551754655-cd27e38d2076?w=600&q=80",
    "Rice":    "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=600&q=80",
    "Wheat":   "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=600&q=80",
    "Soybean": "https://images.unsplash.com/photo-1599940824399-b87987ceb72a?w=600&q=80",
    "Sorghum": "https://images.unsplash.com/photo-1625246333195-78d9c38ad449?w=600&q=80",
    "Potato":  "https://images.unsplash.com/photo-1518977676601-b53f82aba655?w=600&q=80",
}

CROP_DESCS = {
    "Maize":   "High-yield C4 crop. Solar radiation drives productivity.",
    "Rice":    "Warm-season staple. High water demand, NDVI peaks at panicle stage.",
    "Wheat":   "Cool-season cereal. Sensitive to heat stress during grain fill.",
    "Soybean": "Nitrogen-fixing legume. Thrives in warm, well-drained soils.",
    "Sorghum": "Drought-tolerant cereal. Thrives in semi-arid conditions.",
    "Potato":  "Root vegetable. Needs cool temps and consistent soil moisture.",
}

@app.context_processor
def inject_globals():
    return dict(crops=CROPS, form={}, data='{}', data_table=[], stats=None)

_yield_sc = None
_model = None
_wx_sc = None

def get_yield_scaler():
    global _yield_sc
    if _yield_sc is None:
        import joblib
        p = os.path.join(MODEL_DIR, "yield_scaler.pkl")
        if os.path.exists(p):
            _yield_sc = joblib.load(p)
    return _yield_sc

def get_model():
    global _model
    if _model is None:
        p = os.path.join(MODEL_DIR, "fusion_model.keras")
        if os.path.exists(p):
            try:
                _model = keras.models.load_model(p, safe_mode=False)
            except Exception:
                # Fallback: rebuild architecture and load weights only
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from src.fusion_model import build_fusion_model
                img_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS + 2)
                wx_shape  = (SEQUENCE_LENGTH, NUM_WEATHER_FEATURES)
                _model = build_fusion_model(img_shape, wx_shape)
                _model.load_weights(p)
    return _model

def get_wx_scaler():
    global _wx_sc
    if _wx_sc is None:
        import joblib
        p = os.path.join(MODEL_DIR, "weather_scaler.pkl")
        if os.path.exists(p):
            _wx_sc = joblib.load(p)
    return _wx_sc

def load_processed():
    try:
        yields      = np.load(os.path.join(PROCESSED_DIR, "yields.npy"))
        weather     = np.load(os.path.join(PROCESSED_DIR, "weather.npy"))
        ndvi        = np.load(os.path.join(PROCESSED_DIR, "ndvi.npy"))
        crop_labels = np.load(os.path.join(PROCESSED_DIR, "crop_labels.npy"))
        return yields, weather, ndvi, crop_labels
    except Exception:
        return None, None, None, None

@app.route("/")
def home():
    yields, _, _, crop_labels = load_processed()
    stats = None
    if yields is not None:
        names = {0: "Maize", 1: "Rice", 2: "Wheat", 3: "Soybean", 4: "Sorghum", 5: "Potato"}
        crop_avg = {}
        for k, v in names.items():
            mask = crop_labels == k
            crop_avg[v] = round(float(yields[mask].mean()), 2) if mask.any() else 0
        stats = {
            "total_fields": int(len(yields)),
            "avg_yield":    round(float(yields.mean()), 2),
            "max_yield":    round(float(yields.max()),  2),
            "min_yield":    round(float(yields.min()),  2),
            "crop_avg":     crop_avg,
        }
    return render_template("index.html", page="home", stats=stats, data='{}', form={}, data_table=[], crops=CROPS)

@app.route("/eda")
def eda():
    yields, weather, ndvi, crop_labels = load_processed()
    data = {}
    if yields is not None:
        ndvi_spatial = ndvi.mean(axis=(2, 3))
        data["weeks"]     = list(range(1, SEQUENCE_LENGTH + 1))
        data["ndvi_mean"] = ndvi_spatial.mean(axis=0).tolist()
        data["ndvi_std"]  = ndvi_spatial.std(axis=0).tolist()
        names = {0: "Maize", 1: "Rice", 2: "Wheat", 3: "Soybean", 4: "Sorghum", 5: "Potato"}
        data["crop_yields"] = {v: yields[crop_labels == k].tolist() for k, v in names.items()}
        data["crop_means"]  = {v: round(float(yields[crop_labels == k].mean()), 3) for k, v in names.items()}
        wx_mean = weather.mean(axis=1)
        cols = ["Rainfall", "Temperature", "Humidity", "Soil Moisture", "Pesticides", "Year Norm"]
        wx_df = pd.DataFrame(wx_mean, columns=cols[:wx_mean.shape[1]])
        wx_df["Yield"] = yields
        corr = wx_df[["Rainfall", "Temperature", "Humidity", "Soil Moisture", "Yield"]].corr().round(2)
        data["corr_labels"] = corr.columns.tolist()
        data["corr_matrix"] = corr.values.tolist()
    return render_template("index.html", page="eda", data=json.dumps(data), form={}, data_table=[], crops=CROPS)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    form   = {}
    if request.method == "POST":
        form = request.form.to_dict()
        try:
            rainfall   = float(form["rainfall"])
            temp_max   = float(form["temp_max"])
            temp_min   = float(form["temp_min"])
            humidity   = float(form["humidity"])
            solar_rad  = float(form["solar_rad"])
            soil_moist = float(form["soil_moist"])
            peak_ndvi  = float(form["peak_ndvi"])
            mean_ndvi  = float(form["mean_ndvi"])
            crop_name  = form.get("crop_type", "Wheat")
            crop_info  = CROPS.get(crop_name, CROPS["Wheat"])

            model = get_model()
            if model is None:
                result = {"error": "Model not found. Run python src/train.py first."}
            else:
                wx_sc    = get_wx_scaler()
                wx_vals  = np.array([[rainfall / SEQUENCE_LENGTH, temp_max, temp_min,
                                      humidity, solar_rad, soil_moist]], dtype=np.float32)
                wx_input = np.tile(wx_vals, (SEQUENCE_LENGTH, 1)).reshape(1, SEQUENCE_LENGTH, NUM_WEATHER_FEATURES)
                if wx_sc:
                    flat     = wx_input.reshape(-1, NUM_WEATHER_FEATURES)
                    wx_input = wx_sc.transform(flat).reshape(1, SEQUENCE_LENGTH, NUM_WEATHER_FEATURES).astype(np.float32)

                nir_val   = peak_ndvi * 0.8
                red_val   = nir_val * (1 - peak_ndvi) / (1 + peak_ndvi + 1e-8)
                img_input = np.zeros((1, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS + 2), dtype=np.float32)
                img_input[..., 0]  = red_val
                img_input[..., 1]  = nir_val
                img_input[..., -2] = peak_ndvi
                img_input[..., -1] = mean_ndvi

                # Build crop one-hot vector
                crop_order = ["Maize", "Rice", "Wheat", "Soybean", "Sorghum", "Potato"]
                from config import NUM_CROPS
                crop_oh = np.zeros((1, NUM_CROPS), dtype=np.float32)
                if crop_name in crop_order:
                    crop_oh[0, crop_order.index(crop_name)] = 1.0

                raw  = float(model.predict([img_input, wx_input, crop_oh], verbose=0).flatten()[0])
                yield_sc = get_yield_scaler()
                if yield_sc:
                    raw = float(yield_sc.inverse_transform([[raw]])[0][0])
                pred = round(max(0.3, raw), 2)
                result = {
                    "yield": pred,
                    "std":   round(pred * 0.08, 2),
                    "crop":  crop_name,
                    "emoji": crop_info["emoji"],
                }
        except Exception as e:
            import traceback
            result = {"error": str(e) or traceback.format_exc()}
    return render_template("index.html", page="predict", result=result, form=form, data='{}', data_table=[], crops=CROPS, cropImgs=CROP_IMAGES)

@app.route("/comparison")
def comparison():
    data       = {}
    data_table = []
    p = os.path.join(RESULTS_DIR, "all_model_results.csv")
    if os.path.exists(p):
        df         = pd.read_csv(p, index_col=0).round(6)
        data_table = df.reset_index().to_dict(orient="records")
        data = {
            "models": df.index.tolist(),
            "rmse":   df["RMSE"].tolist(),
            "r2":     df["R2"].tolist(),
        }
    return render_template("index.html", page="comparison", data=json.dumps(data), data_table=data_table, form={}, crops=CROPS)

@app.route("/about")
def about():
    return render_template("index.html", page="about", data='{}', form={}, data_table=[], crops=CROPS)

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
