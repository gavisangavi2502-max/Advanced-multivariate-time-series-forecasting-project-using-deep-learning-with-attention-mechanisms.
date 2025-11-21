import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def create_multivariate_windows(df, feature_cols, target_col, window=24):
    data = df[feature_cols + [target_col]].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    X = []
    y = []
    for i in range(len(data_scaled)-window):
        X.append(data_scaled[i:i+window, :-1])
        y.append(data_scaled[i+window, -1])
    return np.array(X), np.array(y), scaler

def train_test_split(X, y, test_ratio=0.2):
    n = len(X)
    split = int(n*(1-test_ratio))
    return X[:split], X[split:], y[:split], y[split:]
