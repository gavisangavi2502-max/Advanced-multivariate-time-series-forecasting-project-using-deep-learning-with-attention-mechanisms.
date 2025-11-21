import numpy as np, matplotlib.pyplot as plt, os
from data_loader import load_csv
from preprocess import create_multivariate_windows, train_test_split
from baseline_lstm import build_lstm
from transformer_model import build_simple_transformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf

def evaluate_model(model, X_test, y_test, horizon=1):
    # For single-step forecast demonstration: X_test provided with windows, predict next step
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    return preds, rmse, mae

def run_training(data_csv, out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    df = load_csv(data_csv)
    feature_cols = ['feat1','feat2','feat3']
    X, y, scaler = create_multivariate_windows(df, feature_cols, 'target', window=48)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)
    # ensure shapes
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # Baseline LSTM
    lstm = build_lstm((X_train.shape[1], X_train.shape[2]), units=64)
    lstm.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=5, batch_size=32)
    preds_lstm, rmse_l, mae_l = evaluate_model(lstm, X_test, y_test)
    # Transformer
    trans = build_simple_transformer((X_train.shape[1], X_train.shape[2]), d_model=64, num_heads=4, ff_dim=128, num_layers=2)
    trans.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=5, batch_size=32)
    preds_tr, rmse_t, mae_t = evaluate_model(trans, X_test, y_test)
    # Save metrics and simple plot
    with open(os.path.join(out_dir,'metrics.txt'),'w') as f:
        f.write(f'LSTM RMSE: {rmse_l:.6f}, MAE: {mae_l:.6f}\n')
        f.write(f'Transformer RMSE: {rmse_t:.6f}, MAE: {mae_t:.6f}\n')
    # Save a plot comparing first 200 points
    plt.figure()
    plt.plot(y_test[:200], label='actual')
    plt.plot(preds_lstm[:200], label='lstm_pred')
    plt.plot(preds_tr[:200], label='trans_pred')
    plt.legend()
    plt.title('Predictions vs Actual (first 200 points)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'predictions_vs_actual.png'))
    plt.close()
    print('Training completed. Outputs in', out_dir)

if __name__=='__main__':
    run_training('data/multivariate_synthetic.csv', out_dir='outputs')
