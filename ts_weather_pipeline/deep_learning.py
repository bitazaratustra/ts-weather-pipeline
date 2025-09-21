import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class PositionalEncoding(layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        position = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
        i = tf.cast(tf.range(self.d_model)[tf.newaxis, :], tf.float32)
        angle_rates = 1 / tf.pow(10000.0, (2 * (i//2)) / tf.cast(self.d_model, tf.float32))
        angle_rads = position * angle_rates
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return inputs + tf.cast(pos_encoding, inputs.dtype)

def transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout=0.1):
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)(x, x)
    attn = layers.Dropout(dropout)(attn)
    out1 = layers.LayerNormalization(epsilon=1e-6)(x + attn)
    ff = layers.Dense(ff_dim, activation="relu")(out1)
    ff = layers.Dense(d_model)(ff)
    ff = layers.Dropout(dropout)(ff)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ff)
    return out2

def build_transformer_encoder(input_shape, output_width, d_model=128, num_heads=4, ff_dim=256, num_blocks=3, dropout=0.1, lr=1e-3):
    seq_len, n_feat = input_shape
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(d_model)(inputs)
    x = PositionalEncoding(d_model)(x)
    for _ in range(num_blocks):
        x = transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(d_model//2, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(output_width)(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model

def make_supervised(df, target_col, input_width, output_width):
    """
    Prepara datos para modelos supervisados (LSTM)
    """
    data = df.select_dtypes(include="number").values
    n_samples = len(data) - input_width - output_width + 1
    X = np.zeros((n_samples, input_width, data.shape[1]))
    y = np.zeros((n_samples, output_width))
    
    for i in range(n_samples):
        X[i] = data[i:i+input_width]
        y[i] = data[i+input_width:i+input_width+output_width, df.columns.get_loc(target_col)]
    
    return X, y, df.columns.tolist()

def build_lstm_model(input_shape, output_width, units=64, lr=1e-3):
    """
    Construye un modelo LSTM
    """
    from tensorflow.keras import layers, models
    
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(units, return_sequences=True)(inputs)
    x = layers.LSTM(units//2)(x)
    x = layers.Dense(units//4, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(output_width)(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                 loss="mse", 
                 metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model