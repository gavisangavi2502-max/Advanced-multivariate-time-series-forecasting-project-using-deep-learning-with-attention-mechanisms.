import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def get_positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2*(i//2))/np.float32(d_model))
    angle_rads = pos * angle_rates
    sines = np.sin(angle_rads[:, 0::2])
    coses = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, coses], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)

def build_simple_transformer(input_shape, d_model=64, num_heads=4, ff_dim=128, num_layers=2):
    seq_len, n_features = input_shape[0], input_shape[1]
    inputs = layers.Input(shape=input_shape)
    # linear projection
    x = layers.Dense(d_model)(inputs)
    # add positional encoding
    pos_enc = get_positional_encoding(seq_len, d_model)
    x = x + pos_enc
    for _ in range(num_layers):
        # Multi-head attention block
        attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        attn_out = layers.Dropout(0.1)(attn_out)
        out1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)
        # Feed-forward
        ff = layers.Dense(ff_dim, activation='relu')(out1)
        ff = layers.Dense(d_model)(ff)
        ff = layers.Dropout(0.1)(ff)
        x = layers.LayerNormalization(epsilon=1e-6)(out1 + ff)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
