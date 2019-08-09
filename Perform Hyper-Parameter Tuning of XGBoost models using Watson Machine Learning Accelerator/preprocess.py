import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

df = pd.read_csv("HIGGS.csv", header=None)

data = df.values

y = data[:,0]
X = data[:,1:]

X_tmp, X_test, y_tmp, y_test = train_test_split(X,y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tmp,y_tmp, random_state=42)

print("Number of features: %d" % (X_train.shape[1]))
print("Number of training   examples: %d" % (X_train.shape[0]))
print("Number of validation examples: %d" % (X_val.shape[0]))
print("Number of test       examples: %d" % (X_test.shape[0]))

dx_train = xgb.DMatrix(X_train, y_train)
dx_val   = xgb.DMatrix(X_val, y_val)
dx_test  = xgb.DMatrix(X_test, y_test)

dx_train.save_binary("train/HIGGS_train.dmatrix")
dx_val.save_binary("val/HIGGS_val.dmatrix")
dx_test.save_binary("test/HIGGS_test.dmatrix")
