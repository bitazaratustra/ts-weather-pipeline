import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def make_supervised(df, target_col, input_width, output_width, step=1, features=None):
    arr = df.copy()
    if features is None:
        features = arr.select_dtypes(include="number").columns.tolist()
    data = arr[features].values
    n = len(data)
    Xs, ys = [], []
    for i in range(0, n - input_width - output_width + 1, step):
        Xs.append(data[i:i+input_width])
        ys.append(arr[target_col].values[i+input_width:i+input_width+output_width])
    return np.array(Xs), np.array(ys), features

def build_lstm_model(input_shape, output_width, units=128, lr=1e-3):
    inputs = layers.Input(shape=input_shape)
    x = layers.Masking()(inputs)
    x = layers.LSTM(units)(x)
    x = layers.Dense(units//2, activation="relu")(x)
    outputs = layers.Dense(output_width)(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model

def build_simple_transformer(input_shape, output_width, d_model=128, num_heads=4, ff_dim=128, lr=1e-3):
    seq_len, n_feat = input_shape
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(d_model)(inputs)
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(d_model)(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(d_model//2, activation="relu")(x)
    outputs = layers.Dense(output_width)(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model
