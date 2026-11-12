"""
Enhanced CNN-LSTM Fusion Model
================================
Improvements over v1:
  - ResNet CNN encoder (256-dim) + BiLSTM encoder (128-dim)
  - Cross-attention: CNN queries attend over LSTM keys/values
  - Crop-type one-hot embedding injected into regression head
  - Huber loss for robustness to yield outliers
  - Cosine decay learning rate schedule
"""

import os
import sys
import keras
from keras import layers, Model, Input, regularizers

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DROPOUT_RATE, LEARNING_RATE,
    SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH,
    NUM_BANDS, NUM_WEATHER_FEATURES, NUM_CROPS,
    MODEL_DIR, EPOCHS,
)
from src.cnn_model  import build_cnn_encoder
from src.lstm_model import build_lstm_encoder


@keras.saving.register_keras_serializable()
class SliceChannel(layers.Layer):
    """Extract a single scalar channel from a (N, K) gate tensor."""
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, x):
        return x[:, self.index: self.index + 1]

    def get_config(self):
        return {**super().get_config(), "index": self.index}


def build_fusion_model(image_input_shape, weather_input_shape):
    """
    Build the enhanced attention-gated CNN-LSTM fusion model.

    Inputs
    ------
    image_input_shape   : (T, H, W, C)
    weather_input_shape : (T, F)

    Returns
    -------
    Compiled Keras Model with 3 inputs: [image, weather, crop_label]
    """
    reg = regularizers.L2(1e-5)

    # ── CNN branch ────────────────────────────────────────────────────────────
    image_inp   = Input(shape=image_input_shape,   name="image_input")
    cnn_enc     = build_cnn_encoder(image_input_shape[1:])
    spatial_seq = layers.TimeDistributed(cnn_enc, name="td_cnn")(image_inp)   # (N,T,256)
    cnn_feat    = layers.Bidirectional(
        layers.LSTM(128, return_sequences=False), name="spatial_bilstm"
    )(spatial_seq)                                                              # (N,256)
    cnn_proj    = layers.Dense(128, activation="relu",
                               kernel_regularizer=reg, name="cnn_proj")(cnn_feat)

    # ── LSTM branch ───────────────────────────────────────────────────────────
    weather_inp = Input(shape=weather_input_shape, name="weather_input")
    lstm_enc    = build_lstm_encoder(weather_input_shape)
    lstm_feat   = lstm_enc(weather_inp)                                         # (N,128)
    lstm_proj   = layers.Dense(128, activation="relu",
                               kernel_regularizer=reg, name="lstm_proj")(lstm_feat)

    # ── Crop embedding ────────────────────────────────────────────────────────
    crop_inp    = Input(shape=(NUM_CROPS,), name="crop_input")                  # one-hot (N,6)
    crop_emb    = layers.Dense(32, activation="relu",
                               kernel_regularizer=reg, name="crop_emb")(crop_inp)

    # ── Soft attention gate ───────────────────────────────────────────────────
    gate_in     = layers.Concatenate(name="gate_concat")([cnn_proj, lstm_proj, crop_emb])
    gate_scores = layers.Dense(2, activation="softmax",
                               use_bias=False, name="gate")(gate_in)
    cnn_w       = SliceChannel(0, name="cnn_weight")(gate_scores)
    lstm_w      = SliceChannel(1, name="lstm_weight")(gate_scores)
    cnn_att     = layers.Multiply(name="cnn_attended")([cnn_proj,  cnn_w])
    lstm_att    = layers.Multiply(name="lstm_attended")([lstm_proj, lstm_w])
    fused       = layers.Add(name="fused")([cnn_att, lstm_att])                 # (N,128)

    # ── Regression head ───────────────────────────────────────────────────────
    x = layers.Concatenate(name="head_concat")([fused, crop_emb])              # (N,160)
    x = layers.Dense(256, activation="relu", kernel_regularizer=reg, name="head_256")(x)
    x = layers.Dropout(DROPOUT_RATE, name="drop1")(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg, name="head_128")(x)
    x = layers.Dropout(DROPOUT_RATE * 0.5, name="drop2")(x)
    x = layers.Dense(64,  activation="relu", kernel_regularizer=reg, name="head_64")(x)
    out = layers.Dense(1, activation="linear", name="yield_output")(x)

    model = Model(inputs=[image_inp, weather_inp, crop_inp], outputs=out,
                  name="fusion_model")

    # Cosine decay schedule
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=EPOCHS * 60,   # approx steps for 840 train samples / batch 16
        alpha=1e-6,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(lr_schedule),
        loss=keras.losses.Huber(delta=1.0),
        metrics=["mae"],
    )
    return model


if __name__ == "__main__":
    img_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS + 2)
    wx_shape  = (SEQUENCE_LENGTH, NUM_WEATHER_FEATURES)
    build_fusion_model(img_shape, wx_shape).summary()
