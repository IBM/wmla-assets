import os
import json
import xgboost as xgb
import argparse
import numpy as np
from sklearn.metrics import mean_squared_error

CLI=argparse.ArgumentParser()
CLI.add_argument("--trainFile", type=str, default="")
CLI.add_argument("--valFile", type=str, default="")
args = CLI.parse_args()

with open("config.json", 'r') as f:
    json_obj = json.load(f)

# Pull parameters from API
learning_rate = float(json_obj["learning_rate"])
num_rounds = int(json_obj["num_rounds"])
max_depth = int(json_obj["max_depth"])
lam = float(json_obj["lambda"])
colsample_bytree = float(json_obj["colsample_bytree"])

# Set params
params = {
  'tree_method': 'gpu_hist',
  'max_depth': max_depth,
  'eta': learning_rate,
  'lambda': lam,
  'colsample_bytree': colsample_bytree
}

# Convert input data from numpy to XGBoost format
dtrain = xgb.DMatrix(args.trainFile)
dval = xgb.DMatrix(args.valFile)

# Train
gbm = xgb.train(params, dtrain, num_rounds)

# Evaluate
true_price = np.expm1(dval.get_label())
pred_price = np.expm1(gbm.predict(dval))
mse_val = mean_squared_error(true_price, pred_price)

# Get output directory for WML-A
dli_result_fs = os.environ['DLI_RESULT_FS']
user = os.environ['USER']
execid = os.environ['DLI_EXECID']
result_dir = "%s/%s/batchworkdir/%s" % (dli_result_fs, user, execid)

# Create output dict
out = []
out.append({
  'mse_val': float(mse_val),
})

# Write output back to WML-A
with open('{}/val_dict_list.json'.format(result_dir), 'w') as f:
    json.dump(out, f)
