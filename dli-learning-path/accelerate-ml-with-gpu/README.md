# use scikit learning，cuML，snapML with Watson Machine Learning Accelerator

The material in this folder supports the scikit-learn/cuML/snapML use cases.  

To run these samples you need have already:

1. Install Watson Machine Learning Accelerator on Cloud Pak for Data 

and then:

2. Create custom anaconda environments(please complete the following steps below).


## Create custom anaconda environments


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

```
bash-4.2# conda create -n rapids-21.06 -c rapidsai -c nvidia -c conda-forge \
    rapids=21.06 python=3.7 cudatoolkit=11.0

```

3. Activate the conda environment:
```
conda activate rapids-21.06 
```

4. Deactivate the conda environment, run:
```
conda deactivate
```

5. Create a conda environment for snapML.

```
(base) bash-4.2# conda create -n snapml-py3.7 python=3.7 

```
6. Activate the conda environment:
```
conda activate snapml-py3.7
```

7. install necessary python packages:
```
pip install snapml==1.7.7rc0
pip install pandas
```

8. Deactivate the conda environment, run:
```
conda deactivate
```

## List of files

| File name | Description |
| --- | --- |

| README.md | Details about how to setup the environment |
| wmla_pod_working.yaml  | yaml file to create a working pod for package installation|
| notebook/KMeans-on-skLearn-cuML.ipynb | Sample to run KMeans on WMLA |
| notebook/LinearRegression-on-skLearn-cuML-snapML.ipynb| Sample to run Linear Regression on WMLA |
| notebook/RandomForest-on-skLearn-cuML-snapML.ipynb | Sample to run Random Forest on WMLA |
| notebook/XGBoost-on-skLearn-cuML-snapML.ipynb| Sample to run XGBoost/SnapBoost on WMLA |
| debug/Debugging-ws.ipynb | Debugging pod |
