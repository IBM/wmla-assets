import xgboost as xgb
import argparse
import numpy as np
from sklearn.metrics import mean_squared_error

CLI=argparse.ArgumentParser()
CLI.add_argument("--trainFile", type=str, default="")
CLI.add_argument("--testFile", type=str, default="")
args = CLI.parse_args()

# Set params
params = {
  'tree_method': 'gpu_hist',
}

# Load training and test data
dtrain = xgb.DMatrix(args.trainFile)
dtest = xgb.DMatrix(args.testFile)

# Training
gbm = xgb.train(params, dtrain)

# Evaluate
true_price = np.expm1(dtest.get_label())
pred_price = np.expm1(gbm.predict(dtest))
mse_test = mean_squared_error(true_price, pred_price)

print("mse_test: %.2f" % (mse_test))
