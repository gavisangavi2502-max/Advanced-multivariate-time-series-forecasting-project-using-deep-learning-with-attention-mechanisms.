# Project Report (Improved Submission)

## Dataset
- Synthetic multivariate dataset (3 features) saved in `data/multivariate_synthetic.csv`.
- In final submission you should replace this with a real dataset (e.g., UCI Electricity, Kaggle energy datasets) and document the source.

## Architecture
- Baseline: LSTM network.
- Advanced: Transformer with positional encodings, layer normalization, multi-head attention, and feed-forward blocks.

## Training & Evaluation
- Train both models for sufficient epochs (example uses 5 for demonstration).
- Evaluate using RMSE and MAE across short/medium/long horizons (implement multi-step rolling prediction).

## Hyperparameter Tuning
- Use grid search or Bayesian optimization (Optuna) for sequence length, d_model, heads, layers, learning rate, dropout.

## Attention Analysis
- Extract attention weights from the Transformer to analyze feature/time-step importance. Visualize using heatmaps and discuss insights.
