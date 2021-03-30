# Build a movie recommendation system with and without GPU in Cloud Pak for Data

The material in this folder supports the movie recommendation use case.  See the full article:  [Build a movie recommendation system with and without GPU](http://)

To build a movie recommendation system, you will need to:

1. Install Watson Machine Learning Accelerator on Cloud Pak for Data 

2. Create a custom conda environment, refer to the steps below.

3. Use the files in this folder to follow the [article](http://).




## Create a custom conda environment

To create a custome conda environment, complete the followin steps:

1. Create a temporary pod using the [wmla_pod_working.yaml](https://raw.githubusercontent.com/IBM/wmla-assets/master/dli-learning-path/datasets/movielens/wmla_pod_working.yaml) file. For additional details on sharing an NFS mount across two persistent volume claims, see: https://docs.openshift.com/container-platform/3.5/install_config/storage_examples/shared_storage.html.
 
 

a. Switch to the WML Accelerator namespace.

b. Create the temporary pod:
```
oc create -f wmla_pod_working.yaml
pod/wmla-working-pod created
```

c. Verify that the pod is in Running state.
```
oc get po |grep wmla-working-pod
wmla-working-pod                                    1/1     Running   0          2m50s
```

d.  Log on to the pod.
```
oc exec -it wmla-working-pod  bash
bash-4.2# 
bash-4.2# cd /opt/anaconda3/
```

2.   Download the [movie_recommendation_env.yml](https://raw.githubusercontent.com/IBM/wmla-assets/master/dli-learning-path/datasets/movielens/movie_recommendation_env.yml) file.

```
(base) bash-4.2# conda install wget
```
```
(base) bash-4.2# wget https://raw.githubusercontent.com/IBM/wmla-assets/master/dli-learning-path/datasets/movielens/movie_recommendation_env.yml
```

3.  Create a conda environment using the movie_recommendation_env.yml file.

a. Create the conda environment.
```
(base) bash-4.2# conda env create -f movie_recommendation_env.yml 

 Uninstalling numpy-1.20.1:
      Successfully uninstalled numpy-1.20.1
Successfully installed absl-py-0.12.0 astunparse-1.6.3 cachetools-4.2.1 chardet-4.0.0 gast-0.3.3 google-auth-1.27.1 google-auth-oauthlib-0.4.3 google-pasta-0.2.0 grpcio-1.36.1 h5py-2.10.0 idna-2.10 importlib-metadata-3.7.3 keras-preprocessing-1.1.2 markdown-3.3.4 numpy-1.18.5 oauthlib-3.1.0 opt-einsum-3.3.0 pyasn1-0.4.8 pyasn1-modules-0.2.8 requests-2.25.1 requests-oauthlib-1.3.0 rsa-4.7.2 scipy-1.4.1 tensorboard-2.4.1 tensorboard-plugin-wit-1.8.0 tensorflow-estimator-2.3.0 tensorflow-gpu-2.3.0 termcolor-1.1.0 urllib3-1.26.3 werkzeug-1.0.1 wrapt-1.12.1 zipp-3.4.1
```

b. Activate the conda environment:
```
conda activate rapids-0.18-movie-recommendation
```

NOTE: To deactive the conda environment, run:
```
conda deactivate
```


## List of files

| File name | Description |
| --- | --- |
| datasets/ml-latest-small.zip | Dataset used by the movie recommendation use case |
| dataset/README.md | Details about the dataset used by the movide recommendation use case |
| notebook/Movie Recommendation Engine - CPU vs GPU.ipynb | Description |
| notebook/Movie-Recommendation-Engine-CPUvsGPU-with-results.ipynb | Description |
| movie_recommendation_env.yml | yml file used to create conda environment |
| wmla_pod_working.yaml  | yaml file used to create temporary pod |
| README.md | Details about movie recomendation use case |
