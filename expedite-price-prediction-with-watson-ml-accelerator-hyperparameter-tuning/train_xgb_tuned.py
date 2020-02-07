import xgboost as xgb
import argparse
import numpy as np
from sklearn.metrics import mean_squared_error

CLI=argparse.ArgumentParser()
CLI.add_argument("--trainFile", type=str, default="")
CLI.add_argument("--testFile", type=str, default="")
args = CLI.parse_args()

# Set params as found by WML-A
params = {
  'tree_method': 'gpu_hist',
  'learning_rate': 0.9597590292372464,
  'num_rounds': 565,
  'max_depth': 13,
  'lambda': 1584.7191653582931, 
  'colsample_bytree': 0.47
}

# Load training and test data
dtrain = xgb.DMatrix(args.trainFile)
dtest = xgb.DMatrix(args.testFile)

# Training
gbm = xgb.train(params, dtrain, params['num_rounds'])

# Evaluate
true_price = np.expm1(dtest.get_label())
pred_price = np.expm1(gbm.predict(dtest))
mse_test = mean_squared_error(true_price, pred_price)

# Output
print("mse_test: %.2f" % (mse_test))


