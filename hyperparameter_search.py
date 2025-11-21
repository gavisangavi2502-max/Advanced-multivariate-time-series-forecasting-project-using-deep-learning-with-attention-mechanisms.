# A tiny example of grid-search over two hyperparameters (d_model and num_heads)
import itertools, os
from transform_model_stub import transform_stub
# This is a placeholder to show how to structure a tuning run.
# For the real project, replace transform_stub with training loops and return metrics.
def run_grid():
    d_models = [32,64]
    heads = [2,4]
    results = []
    for d,h in itertools.product(d_models, heads):
        # pretend we trained and obtained a RMSE
        rmse = 1.0/(d*h/64)
        results.append({'d_model':d, 'num_heads':h, 'rmse':rmse})
    print(results)
if __name__=='__main__':
    run_grid()
