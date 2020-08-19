# Expedite Retail Price Prediction with Watson ML Accelerator Hyperparameter Optimization  

The adoption of AI has been increasing steadily across all business sectors as more industry leaders understand the value that data and ML models can bring to their business. Some of the benefits that cut across many economy sectors are (a) lower operational costs due to process automation, (b) higher revenues thanks to better productivity and enhanced user experiences and (c) better compliance and reinforced security. 

In Retail in particular, AI can provide benefits with optimization, automation and scale. Retailers today are using data to understand customers and enhance existing offerings to differentiate from the competition. They can also better understand shopping behavior data, anticipate customer needs and interests, and respond with personalized offers and experiences to increase the effectiveness of their promotions, and boost sales.

One key workload for every retailer is Price Optimization, namely the determination of a suitable offering price for a particular item. The opportunity that AI brings here is optimization across a wide assortment of items based on a variety of factors. AI models can be used to determine the best price for each item using data on seasonality along with real-time inputs on inventory levels and competitive products and prices.  AI can also show retailers likely outcomes of different pricing strategies so they can come up with the best promotional offers, acquire more customers, and increase sales. 

In order to realize the potential benefits outlined above, it is crucial to design and deploy ML models that are able to predict the most suitable price for each item with the highest possible accuracy. It is widely acknowledged today that Gradient Boosting Machine (GBM) is among the most powerful ML models, offering the highest generalization accuracy for most tasks that involve tabular datasets. With that motivation, this tutorial is focused on GBM as the ML model of choice for the Price Optimization task. We choose a public dataset from kaggle, namely from the Mercari price optimization competition, which can be found at https://www.kaggle.com/saitosean/mercari. We then make use of a popular GBM model, XGBoost (https://github.com/dmlc/xgboost). In order to achieve good generalization accuracy with XGBoost we perform Hyper-parameter Tuning (HPT), i.e. try different hyper-parameter sets and select the ones that give the best accuracy in a validation dataset. To do that, we use the Watson Machine Learning Accelerator (WMLA) suite, which is a resource orchestrator and task scheduler and can seamlessly distribute the HPT task across a cluster of nodes and GPUs. 

This tutorial will demonstrate the value proposition of WMLA, i.e., ease of use and high resource efficiency for distributed ML jobs, as well as the power of the HPT process, which produces an XGBoost model of higher generalization accuracy in unseen data.


## Prepare cluster for HPO job submission - Create conda environment (on all nodes)


### On a POWER cluster:
Create a conda env and installed pre-build XGBoost (as egoadmin user)
```
conda create --name dli-xgboost python=3.6 py-xgboost-gpu
```

### On an x86 cluster

Compile XGBoost:
```
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost/
mkdir build
cd build
PATH=/opt/wmla-ig/anacondapowerai161-1/anaconda/pkgs/cudatoolkit-dev-10.1.168-513.g7069ee4/bin:$PATH
cmake3 .. -DUSE_CUDA=ON
make -j
```

Create conda env and install XGBoost into it:
```
conda create --name dli-xgboost --yes pip python=3.7
conda activate dli-xgboost
conda install numpy scikit-learn scipy
python setup.py install
source deactivate
```


## Create XGBoost BYOF plugin (on management node only)

```
export DLI_EGO_TOP=/opt/ibm/spectrumcomputing
```

Check the value of `DL_NFS_PATH` 
```
$ cat $DLI_EGO_TOP/dli/conf/dlpd/dlpd.conf | grep DL_NFS_PATH
    "DL_NFS_PATH": "/dlishared",
```
```
export DL_NFS_PATH=/dlishared
```

Create a file called `XGboost.conf` with the following contents:
```
{

    "desc" : 
    [{" ": "XGboost. Currently in development phase."},
     {" ": "Examples:"},
     {" ": "$ python dlicmd.py --exec-start XGboost <connection-options> --ig <ig> --model-main XGboost_Main.py"}
    ],

    "deployMode Desc": "Optional",
    "deployMode": "cluster",

    "appName Desc" : "This is required",
    "appName": "dlicmdXGboost",

    "numWorkers Desc": "Optional number of workers",
    "numWorkers": 1,

    "maxWorkers Desc": "User can't specify more than this number",
    "maxWorkers": 1,

    "maxGPUPerWorker Desc": "User can't specify more than this number",
    "maxGPUPerWorker": 10,

    "egoSlotRequiredTimeout Desc": "Optional",
    "egoSlotRequiredTimeout": 120,

    "workerMemory Desc" : "Optional",
    "workerMemory": "2G",

    "frameworkCmdGenerator Desc": "",
    "frameworkCmdGenerator": "XGboostCmdGen.py"
}
```
Move it to the following directory:
```
sudo mv XGboost.conf $DLI_EGO_TOP/dli/conf/dlpd/dl_plugins
```
Create a file called `XGboost_wrapper.sh` with the following contents:
```
#!/bin/sh
source activate dli-xgboost
python $@
```
and create a file called `XGboostCmdGen.py` with the following contents:
```
#!/usr/bin/env python2
import os.path, sys
from os import environ

"""
"""

def main():

   cmd = ""

   if "DLI_SHARED_FS" in os.environ:
      print (environ.get('DLI_SHARED_FS'))
      cmd = environ.get('DLI_SHARED_FS') + "/tools/spark_tf_launcher/launcher.py"
   else:
      print("Error: environment variable DLI_SHARED_FS must be defined")
      sys.exit()

   if "APP_NAME" in os.environ:
      cmd = cmd + " --sparkAppName=" + environ.get('APP_NAME')
   else:
      print("Error: environment variable APP_NAME must be defined")
      sys.exit()

   if "MODEL" in os.environ:
      cmd = cmd + " --model=" + environ.get('MODEL')
   else:
      print("Error: environment variable MODEL must be defined")
      sys.exit()

   if "REDIS_HOST" in os.environ:
      cmd = cmd + " --redis_host=" + environ.get('REDIS_HOST')
   else:
      print("Error: environment variable REDIS_HOST must be defined")
      sys.exit()

   if "REDIS_PORT" in os.environ:
      cmd = cmd + " --redis_port=" + environ.get('REDIS_PORT')
   else:
      print("Error: environment variable REDIS_PORT must be defined")
      sys.exit()

   if "GPU_PER_WORKER" in os.environ:
      cmd = cmd + " --devices=" + environ.get('GPU_PER_WORKER')
   else:
      print("Error: environment variable GPU_PER_WORKER must be defined")
      sys.exit()

   cmd = cmd + " --work_dir=" + os.path.dirname(environ.get('MODEL'))
   cmd = cmd + " --app_type=executable"
   
   cmd = cmd + " --model=" + environ.get('DLI_SHARED_FS') + "/tools/dl_plugins/XGboost_wrapper.sh --"
   cmd = cmd + " " + environ.get('MODEL')

   # adding user args
   for i in range(1, len(sys.argv)):
      cmd += " " + sys.argv[i]

   # Expected result in json
   print('{"CMD" : "%s"}' % cmd)

if __name__ == "__main__":
   sys.exit(main())
```
Move those files and make them executable:
```
sudo mv XGboost_wrapper.sh $DL_NFS_PATH/tools/dl_plugins 
sudo mv XGboostCmdGen.py $DL_NFS_PATH/tools/dl_plugins 
sudo chmod +x $DL_NFS_PATH/tools/dl_plugins/XGboost_wrapper.sh
sudo chmod +x $DL_NFS_PATH/tools/dl_plugins/XGboostCmdGen.py
```

## Download and prepare dataset
```
mkdir $DL_NFS_PATH/datasets/higgs
cd $DL_NFS_PATH/datasets/higgs
wget --no-check-certificate https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
gunzip HIGGS.csv.gz 
mkdir train
mkdir val
mkdir test
```
Create the following python script (`preprocess.py`) in the current folder:
```python
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
```
Execute the pre-processing script to generate the data files:
```bash
conda activate dli-xgboost
conda install pandas
python preprocess.py
```
You should see the following output:
```bash
Number of features: 28
Number of training   examples: 6187500
Number of validation examples: 2062500
Number of test       examples: 2750000
```

Check the value of `DLI_DATA_FS` 
```
$ cat $DLI_EGO_TOP/dli/conf/dlpd/dlpd.conf | grep DLI_DATA_FS
    "DLI_DATA_FS": "/dlidata/",
```

Copy the train and validation dataset to 'DLI_DATA_FS'

```
$ pwd
/dlidata/dataset/price_prediction
$ ll -lt
-rw-rw-r-- 1 egoadmin egoadmin 210025976 Nov  7 14:38 pp_val.dmatrix
-rw-rw-r-- 1 egoadmin egoadmin 630075452 Nov  7 14:38 pp_train.dmatrix
```

## Run XGBoost with default parameters
Create the following file `train_xgb_default.py`:
```python
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import argparse

CLI=argparse.ArgumentParser()
CLI.add_argument("--trainFile", type=str, default="")
CLI.add_argument("--valFile", type=str, default="")
CLI.add_argument("--testFile", type=str, default="")
args = CLI.parse_args()

# Set params
params = {
  'tree_method': 'gpu_hist',
  'max_bin': 64,
  'objective': 'binary:logistic',
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
gbm = xgb.train(params, dtrain)

# Inference
p1_train = gbm.predict(dtrain)
p1_dev  = gbm.predict(ddev)
p1_test  = gbm.predict(dtest)

# Evaluate
auc_train = roc_auc_score(y_train, p1_train)
auc_dev = roc_auc_score(y_dev, p1_dev)
auc_test = roc_auc_score(y_test, p1_test)

print("auc_test: %f, auc_val: %f, auc_test: %f" % (auc_train, auc_dev, auc_test))
```

Run the model with default parameter (using the `dli-xgboost` environment as before):
```bash
(dli-xgboost)# python train_xgb_default.py --trainFile  /dlidata/dataset/price_prediction/pp_train.dmatrix --testFile /dlidata/dataset/price_prediction/pp_val.dmatrix
[04:16:57] 833433x93 matrix with 77509269 entries loaded from /dlidata/dataset/price_prediction/pp_train.dmatrix
[04:16:57] 277812x93 matrix with 25836516 entries loaded from /dlidata/dataset/price_prediction/pp_val.dmatrix
mse_test: 1231.55
```

## Tuning XGboost with Watson ML Accelerator Hyperparameter Optimization (HPO)
Let's see if we can do better with Watson ML Accelerator HPO.   Download the notebook `XGBoost tuning demo.ipynb` and open the notebook with your favorite tool.

Install and Configure Watson ML Accelerator by executing Step 1 - 4 of the runbook: https://github.com/IBM/wmla-assets/blob/master/runbook/WMLA_installation_configuration.md

Update first cell of the notebook, including:
```
- hostname
- username, password
- protocol (http or https)
- http or https port
- sigName
- Dataset location
```

Update second cell of the notebook, including:
```
- maxJobNum:  total number of tuning jobs to be running
- maxParalleJobNum:  total number of tuning jobs running in parallel, which is equivalent to total number of GPUs available in the cluster
```

In this notebook we will tune five parameters of XGBoost model.   Execute the notebook to kick off your parallel model tuning jobs!!!  

Execute the fourth cell to monitor the job progress.   The recommended optimal set of parameter with the best metric will be returned. 

```
....
Hpo task Admin-hpo-83966261958354 state RUNNING progress 56%
Hpo task Admin-hpo-83966261958354 completes with state FINISHED
{
    "best": {
        "appId": "Admin-84189701779622-1370733872",
        "driverId": "driver-20191108160851-0342-bacfbcb3-ed76-4f70-92f5-65062f92d1cb",
        "endTime": "2019-11-08 16:11:07",
        "hyperParams": [
            {
                "dataType": "double",
                "fixedVal": "0.9597590292372464",
                "name": "learning_rate",
                "userDefined": false
            },
            {
                "dataType": "int",
                "fixedVal": "565",
                "name": "num_rounds",
                "userDefined": false
            },
            {
                "dataType": "int",
                "fixedVal": "13",
                "name": "max_depth",
                "userDefined": false
            },
            {
                "dataType": "double",
                "fixedVal": "1584.7191653582931",
                "name": "lambda",
                "userDefined": false
            },
            {
                "dataType": "double",
                "fixedVal": "0.47",
                "name": "colsample_bytree",
                "userDefined": false
            }
        ],
        "id": 6,
        "maxiteration": 0,
        "metricVal": 1036.0960693359375,
        "startTime": "2019-11-08 16:08:51",
        "state": "FINISHED"
    },
    ......
```


Create the following file `train_xgb_tuned.py` with parameters returned from tuning jobs
```
import xgboost as xgb
import argparse
import numpy as np
from sklearn.metrics import mean_squared_error

CLI=argparse.ArgumentParser()
CLI.add_argument("--trainFile", type=str, default="")
CLI.add_argument("--testFile", type=str, default="")
args = CLI.parse_args()

# Set params as found by WML-A HPO
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
```

## Run XGBoost with tuned parameters
```
(dli-xgboost)# python train_xgb_tuned.py --trainFile  /dlidata/dataset/price_prediction/pp_train.dmatrix --testFile /dlidata/dataset/price_prediction/pp_val.dmatrix
[04:23:26] 833433x93 matrix with 77509269 entries loaded from /dlidata/dataset/price_prediction/pp_train.dmatrix
[04:23:26] 277812x93 matrix with 25836516 entries loaded from /dlidata/dataset/price_prediction/pp_val.dmatrix
mse_test: 1036.10
```

## Conclusion

Mean Squared Error (MSE)  measures the average of the squares of the errors,  which is the average squared difference between the estimated values and the actual value. 

The lower the value of MSE represents smaller approximation error and delivers better generalization accuracy.     

In our experiment the MSE value with WMLA HPO tuned parameter is 1036.10,  which delivers better generalization accuracy by comparing with default parameter of MSE value 1231.55.

In this tutorial we demonstrate Watson ML Accelerator ease of use and efficiency of automating parallel HyperParameter Tuning jobs,  and delivers more accurate retail price prediction with better generalization accuracy of XGBoost model 

