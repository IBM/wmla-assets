# use scikit learning and cuML with Watson Machine Learning Accelerator

The material in this folder supports the scikit-learn/cuML/snapML use case.  

to run these samples you need to:

1. Install Watson Machine Learning Accelerator on Cloud Pak for Data 

2. Create custom conda environments, refer to the steps below.

3. Enlarge the wmla working pod's memory, refer to the steps below.



## Create a custom conda environment

To create a custome conda environment, complete the following steps:

1. Create a temporary pod using wmla_pod_working.yaml
   (https://raw.githubusercontent.com/IBM/wmla-assets/dli-learning-path/accelerate-ml-with-gpu/wmla_pod_working.yaml). 

a. Switch to the WML Accelerator namespace.
```
oc project <wml-accelerator-ns>
```

b. Create the temporary pod:
```
oc create -f wmla_pod_working.yaml
```

c. Verify that the pod is in Running state.
```
oc get po |grep wmla-working-pod
```

d.  Log on to the pod.
```
oc exec -it wmla-working-pod -- bash
bash-4.2# source /opt/anaconda3/etc/profile.d/conda.sh
```

2. Create a conda environment for cuML.

a. Create the conda environment for cuML:
```
bash-4.2# conda create -n rapids-21.06 -c rapidsai -c nvidia -c conda-forge \
    rapids=21.06 python=3.7 cudatoolkit=11.0

```

b. Activate the conda environment:
```
conda activate rapids-21.06 
```

NOTE: To deactivate the conda environment, run:
```
conda deactivate
```

3. Create a conda environment for snapML.

a. Create the conda environment for snapML:
```
(base) bash-4.2# conda create -n snapml-py3.7 python=3.7 snapml=1.7.6

```

b. Activate the conda environment:
```
conda activate snapml-py3.7
```

NOTE: To deactivate the conda environment, run:
```
conda deactivate
```

c. install necessary python packages:
```
pip install snapml
pip install pandas
```

## enlarge the wmla working pod's memory

1. logon to the wmla dlpd pod
a. Switch to the WML Accelerator namespace.
```
oc project <wml-accelerator-ns>
```

b. find the wmla-dlpd pod.
```
oc get po |grep wmla-dlpd
```

d.  Log on to the pod.
```
oc exec -it <wmla-dlpd-pod-name> -c dlpd -- bash
```
2. enlarge the value of TASK12N memory in dlpd.conf
a. find the file dlpd.conf and check its parameters
```
bash-4.2# grep MSD_TASK12N_MEMORY /var/shareDir/dli/conf/dlpd/dlpd.conf
    "MSD_TASK12N_MEMORY": "4G",
    "MSD_TASK12N_MEMORY_EDT": "8G",
    "MSD_TASK12N_MEMORY_LIMIT_EDT": "16G",
```
b. modify the value of parameters MSD_TASK12N_MEMORY
```
bash-4.2# vim /var/shareDir/dli/conf/dlpd/dlpd.conf
    "MSD_TASK12N_MEMORY": "32G",
    "MSD_TASK12N_MEMORY_EDT": "32G",
    "MSD_TASK12N_MEMORY_LIMIT_EDT": "32G",
```
c. exit the pod
```
bash-4.2# exit
```
3. restart wmla-dlpd pod to make parameter take effect.
a. delete original wmla-dlpd pod
```
oc delete po <wmla-dlpd-pod-name> 
```
b. wait the new pod startup.
```
oc get po |grep wmla-dlpd
wmla-dlpd-8dcc84l1-8ju78            2/2     Running   0          3m20s
```

## List of files

| File name | Description |
| --- | --- |

| README.md | Details about how to setup the environment |
| wmla_pod_working.yaml  | yaml file used to create temporary pod |
| notebook/KMeans-on-skLearn-cuML.ipynb | Sample to run KMeans on WMLA |
| notebook/LinearRegression-on-skLearn-cuML-snapML.ipynb| Sample to run Linear Regression on WMLA |
| notebook/RandomForest-on-skLearn-cuML-snapML.ipynb | Sample to run Random Forest on WMLA |
| notebook/XGBoost-on-skLearn-cuML.ipynb| Sample to run XGBoost on WMLA |
| notebook/Debugging-ws.ipynb | Debughinh pod |
