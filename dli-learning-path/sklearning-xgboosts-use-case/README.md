# use scikit learning, cuML and xgboost with Watson Machine Learning Accelerator

The material in this folder supports the scikit learning use case and cuML use case.  

To build a movie recommendation system, you will need to:

1. Install Watson Machine Learning Accelerator on Cloud Pak for Data 

2. Create a custom conda environment, refer to the steps below.

3. Use the files in this folder to follow the [article](http://).




## Create a custom conda environment

To create a custome conda environment, complete the following steps:

1. Create a temporary pod using the [wmla_pod_working.yaml](https://raw.githubusercontent.com/IBM/wmla-assets/_________/wmla_pod_working.yaml) file. For additional details, see: https://docs.openshift.com/container-platform/3.5/install_config/storage_examples/shared_storage.html.
 

a. Switch to the WML Accelerator namespace.
```
oc project wml-accelerator
```

b. Create the temporary pod:
```
oc create -f wmla_pod_working.yaml
pod/wmla-working-pod created
```

c. Verify that the pod is in Running state.
```
oc get po |grep wmla-working-pod
wmla-working-pod                                    1/1     Running   0          2m50s
```

d.  Log on to the pod.
```
oc exec -it wmla-working-pod -- bash
bash-4.2# 
bash-4.2# cd /opt/anaconda3/
```

2.  Create a conda environment using the movie_recommendation_env.yml file.

a. Create the conda environment.
```
(base) bash-4.2# conda create -n rapids-21.06 -c rapidsai -c nvidia -c conda-forge \
    rapids=21.06 python=3.7 cudatoolkit=11.0


(base) bash-4.2# conda install -c rapidsai -c nvidia -c conda-forge -n rapids-21.06 rapids=21.06

```

b. Activate the conda environment:
```
conda activate rapids-21.06 
```

NOTE: To deactivate the conda environment, run:
```
conda deactivate
```


## List of files

| File name | Description |
| --- | --- |
| dataset/ml-latest-small.zip | Dataset used by the scikit learning, cuML and xgboost use case |
| wmla_pod_working.yaml  | yaml file used to create temporary pod |
| notebook/cuML-KMeans-on-wmla.ipynb | Sample to run KMeans on WMLA |
| notebook/cuML-LinearRegression-on-wmla.ipynb | Sample to run Linear Regression on WMLA |
| notebook/cuML-RandomForest-on-wmla.ipynb | Sample to run Random Forest on WMLA |
| notebook/xgboost-on-wmla.ipynb | Sample to run XGBoost on WMLA |
| README.md | Details about scikit learning and cuML use case |
