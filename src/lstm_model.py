"""
Stacked Bidirectional LSTM encoder for weather time-series.
"""
import os
import sys
import keras
from keras import layers, Model, Input

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LSTM_UNITS, DROPOUT_RATE, SEQUENCE_LENGTH, NUM_WEATHER_FEATURES


def build_lstm_encoder(input_shape):
    """
    Stacked BiLSTM encoder with layer normalization.
    input_shape: (T, F)  →  outputs (128,) feature vector.
    """
    inp = Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inp)

    # Layer 1 — BiLSTM, return sequences for stacking
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS[0], return_sequences=True, dropout=DROPOUT_RATE * 0.5)
    )(x)
    x = layers.LayerNormalization()(x)

    # Layer 2 — BiLSTM, collapse to single vector
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS[1], return_sequences=False, dropout=DROPOUT_RATE * 0.5)
    )(x)
    x = layers.LayerNormalization()(x)

    x = layers.Dropout(DROPOUT_RATE)(x)
    out = layers.Dense(128, activation="relu")(x)

    return Model(inp, out, name="lstm_encoder")


def build_standalone_lstm(weather_input_shape):
    """
    Full LSTM model with regression head.
    weather_input_shape: (T, F)
    """
    inp = Input(shape=weather_input_shape, name="weather_input")
    lstm_enc = build_lstm_encoder(weather_input_shape)
    x = lstm_enc(inp)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    out = layers.Dense(1, activation="linear", name="yield_output")(x)

    model = Model(inp, out, name="standalone_lstm")
    model.compile(optimizer=keras.optimizers.Adam(5e-4),
                  loss=keras.losses.Huber(delta=1.0), metrics=["mae"])
    return model


if __name__ == "__main__":
    shape = (SEQUENCE_LENGTH, NUM_WEATHER_FEATURES)
    build_standalone_lstm(shape).summary()
