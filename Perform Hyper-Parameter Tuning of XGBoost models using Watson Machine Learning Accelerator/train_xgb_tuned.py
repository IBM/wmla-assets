CLI.add_argument("--valFile", type=str, default="")
CLI.add_argument("--testFile", type=str, default="")
args = CLI.parse_args()

# Set params
params = {
  'tree_method': 'gpu_hist',
  'max_bin': 64,
  'objective': 'binary:logistic',
  'learning_rate': 0.6181877638234499,
  'num_rounds': 491,
  'max_depth':13,
  'lambda':51762.958983821656,
  'colsample_bytree':0.33,
}

# Load data
dtrain = xgb.DMatrix(args.trainFile)
ddev = xgb.DMatrix(args.valFile)
dtest = xgb.DMatrix(args.testFile)

# Get labels
y_train = dtrain.get_label()
y_dev = ddev.get_label()
y_test = dtest.get_label()

# Train
gbm = xgb.train(params, dtrain, params['num_rounds'])

# Inference
p1_train = gbm.predict(dtrain)
p1_dev  = gbm.predict(ddev)
p1_test  = gbm.predict(dtest)

# Evaluate
auc_train = roc_auc_score(y_train, p1_train)
auc_dev = roc_auc_score(y_dev, p1_dev)
auc_test = roc_auc_score(y_test, p1_test)

print("auc_train: %f, auc_val: %f, auc_test: %f" % (auc_train, auc_dev, auc_test))
