import os
import json
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import argparse

CLI=argparse.ArgumentParser()
CLI.add_argument("--trainFile", type=str, default="")
CLI.add_argument("--valFile", type=str, default="")
CLI.add_argument("--testFile", type=str, default="")
args = CLI.parse_args()



with open("config.json", 'r') as f:
    json_obj = json.load(f)

# pull parameters from API
learning_rate = float(json_obj["learning_rate"])
num_rounds = int(json_obj["num_rounds"])
max_depth = int(json_obj["max_depth"])
lam = float(json_obj["lambda"])
colsample_bytree = float(json_obj["colsample_bytree"])

# Set params
params = {
  'max_depth': max_depth,
  'eta': learning_rate,
  'tree_method': 'gpu_hist',
  'max_bin': 64,
  'objective': 'binary:logistic',
  'lambda': lam,
  'colsample_bytree': colsample_bytree,
}

# Convert input data from numpy to XGBoost format
dtrain = xgb.DMatrix(args.trainFile)
ddev = xgb.DMatrix(args.valFile)
dtest = xgb.DMatrix(args.testFile)

y_train = dtrain.get_label()
y_dev = ddev.get_label()
y_test = dtest.get_label()

# Train
gbm = xgb.train(params, dtrain, num_rounds)

# Inference
p1_train = gbm.predict(dtrain)
p1_dev  = gbm.predict(ddev)
p1_test  = gbm.predict(dtest)

# Evaluate
auc_train = roc_auc_score(y_train, p1_train)
auc_dev = roc_auc_score(y_dev, p1_dev)
auc_test = roc_auc_score(y_test, p1_test)

dli_result_fs = os.environ['DLI_RESULT_FS']
user = os.environ['USER']
execid = os.environ['DLI_EXECID']
result_dir = "%s/%s/batchworkdir/%s" % (dli_result_fs, user, execid)

out = []
out.append({
    'auc_train': auc_train,
    'auc_dev': auc_dev,
    'auc_test': auc_test
    })

with open('{}/val_dict_list.json'.format(result_dir), 'w') as f:
    json.dump(out, f)
