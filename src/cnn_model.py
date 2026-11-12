"""
ResNet-style CNN encoder for Sentinel-2 spatial feature extraction.
Residual connections prevent gradient vanishing on deeper architectures.
"""
import os
import sys
import keras
from keras import layers, Model, Input

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CNN_FILTERS, DROPOUT_RATE, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS


def _residual_block(x, filters, stride=1):
    """Pre-activation residual block (He et al. 2016)."""
    shortcut = x
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)

    # Project shortcut if shape changes
    if shortcut.shape[-1] != filters or stride > 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
    return layers.Add()([x, shortcut])


def build_cnn_encoder(input_shape):
    """
    ResNet-style CNN spatial feature extractor for a single timestep.
    input_shape: (H, W, C)  →  outputs (256,) feature vector.
    """
    inp = Input(shape=input_shape)

    # Stem
    x = layers.Conv2D(CNN_FILTERS[0], 3, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Stage 1 — 2 residual blocks, stride-2 downsample
    x = _residual_block(x, CNN_FILTERS[0])
    x = _residual_block(x, CNN_FILTERS[1], stride=2)   # 32→16

    # Stage 2 — 2 residual blocks, stride-2 downsample
    x = _residual_block(x, CNN_FILTERS[1])
    x = _residual_block(x, CNN_FILTERS[2], stride=2)   # 16→8

    # Stage 3 — 1 residual block
    x = _residual_block(x, CNN_FILTERS[2])

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    out = layers.Dense(256, activation="relu")(x)

    return Model(inp, out, name="cnn_encoder")


def build_standalone_cnn(image_input_shape):
    """
    Full CNN model: TimeDistributed ResNet encoder → BiLSTM → regression head.
    image_input_shape: (T, H, W, C)
    """
    inp = Input(shape=image_input_shape, name="image_input")
    cnn_enc = build_cnn_encoder(image_input_shape[1:])
    x = layers.TimeDistributed(cnn_enc)(inp)                    # (T, 256)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(DROPOUT_RATE * 0.5)(x)
    out = layers.Dense(1, activation="linear", name="yield_output")(x)

    model = Model(inp, out, name="standalone_cnn")
    model.compile(optimizer=keras.optimizers.Adam(5e-4),
                  loss=keras.losses.Huber(delta=1.0), metrics=["mae"])
    return model


if __name__ == "__main__":
    shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS + 2)
    build_standalone_cnn(shape).summary()
