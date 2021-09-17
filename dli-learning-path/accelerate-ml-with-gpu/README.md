# Apply use case of scikit learning，cuML，snap ML with Watson Machine Learning Accelerator

The material in this folder supports the scikit-learn/cuML/snapML use cases.  

To run these samples you must:

1. Install Watson Machine Learning Accelerator on Cloud Pak for Data, see (https://www.ibm.com/docs/en/cloud-paks/cp-data/4.0?topic=iwmla-installing-watson-machine-learning-accelerator).

2. After  Watson Machine Learning Accelerator is installed, create a custom Anaconda environment using the intructions below.


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

## Add the notebook sample

To create the notebook file in IBM Cloud Pak for Data: 

1.	From your project, click **Add to Project** > **Notebook**.
2.	On the **New Notebook** page, click the From **URL method** to use to create your notebook. 
![add-git-sample](https://user-images.githubusercontent.com/29407430/133499710-84b0b8e2-63ed-4d59-bfc6-82bf86e6400f.jpg)

From the provided samples, input the URL:

For Linear Regression (Generated Data), use: https://github.com/IBM/wmla-assets/blob/master/dli-learning-path/accelerate-ml-with-gpu/notebook/LinearRegression-on-skLearn-cuML-snapML-GeneData.ipynb
For Linear Regression (Price Data), use: https://github.com/IBM/wmla-assets/blob/master/dli-learning-path/accelerate-ml-with-gpu/notebook/LinearRegression-on-skLearn-cuML-snapML-PriceData.ipynb
For Random Forest, use:	https://github.com/IBM/wmla-assets/blob/master/dli-learning-path/accelerate-ml-with-gpu/notebook/RandomForest-on-skLearn-cuML-snapML.ipynb
For XGBoost/SnapBoost, use:	https://github.com/IBM/wmla-assets/blob/master/dli-learning-path/accelerate-ml-with-gpu/notebook/XGBoost-on-skLearn-cuML-snapML.ipynb

3. Click **Create Notebook**. The notebook opens in edit mode and is locked by you. 

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
