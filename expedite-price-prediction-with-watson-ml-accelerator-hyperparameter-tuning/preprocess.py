import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import xgboost as xgb

# read tsv file
df = pd.read_csv('train.tsv',sep="\t", index_col='train_id')

# some basic filtering
df = df[df['price']>0].fillna("null")

# text features
textual = ['name', 'category_name', 'item_description']

# categorical features
categoricals = ['item_condition_id', 'brand_name', 'shipping']

# target encoding of categoricals
for col in categoricals:
    df[col] = df[col].map(df.groupby(col)['price'].mean())
    
# extract labels
y = np.log1p(df['price'].values)

# extract categoricals
X = df[categoricals].values

# tfidf encoding of textual features
for col in textual:
    x_add = TfidfVectorizer(max_features=30, stop_words='english').fit_transform(df[col])
    X = np.hstack((X, x_add.toarray()))

# trainval/test split
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=42)

# train/val split
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, random_state=42)

# convert to XGB data structure
dtrainval = xgb.DMatrix(X_trainval, y_trainval)
dtrain = xgb.DMatrix(X_train, y_train)
dval = xgb.DMatrix(X_val, y_val)
dtest = xgb.DMatrix(X_test, y_test)

# write to binary files
dtrainval.save_binary("pp_trainval.dmatrix")
dtrain.save_binary("pp_train.dmatrix")
dval.save_binary("pp_val.dmatrix")
dtest.save_binary("pp_test.dmatrix")

'''
params = {
    'tree_method': 'gpu_hist'
}

bst = xgb.train(params, dtrain)
z_train = bst.predict(dtrain)
z_val = bst.predict(dval)
z_test = bst.predict(dtest)

def rmspe(a, b):
    return np.sqrt(np.mean((b/a-1) ** 2))

rmspe_train = rmspe(np.expm1(y_train), np.expm1(z_train))
rmspe_val = rmspe(np.expm1(y_val), np.expm1(z_val))
rmspe_test = rmspe(np.expm1(y_test), np.expm1(z_test))
'''
