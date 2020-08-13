


## Summary
&nbsp;
&nbsp;
Dask is a flexible library for parallel computing in Python.    Dask is Python friendly by using Python APIs and supporting data structures such as Numpy and Pandas.    Dask represents parallel computations with task graphs.   Dask Distributed Scheduler could scale up to multiple nodes cluster,  distributes and executes these task graphs in the cluster.


Elastic Distributed Dask of Watson Machine Learning Accelerator supports dynamic scaling of Dask Cluster by adding new Dask-CUDA-worker,  according to incoming workload demand.       The Dask Cluster starts with minimal resources (1 GPU per worker) .     Spectrum Conductor monitors incoming workload demand,  and dynamically adds new workers to support increasing workloads.     Spectrum Conductor also monitors resource utilization per worker,  and gracefully pre-empties idle worker(s) from the cluster .


&nbsp;
&nbsp;

## Description
In this learning module you will understand how to:
- Install and Deploy Elastic Distributed Dask Cluster 
- Start Elastic Distributed Dask Cluster with 1 GPU
- Experience auto scaling up of Elastic Distributed Dask Cluster by executing notebook to distribute xgboost model training with multiple Dask-CUDA-workers (worker per GPU)
- Experience auto scaling down of Elastic Distributed Dask Cluster


&nbsp;
&nbsp;

1. Install and Deploy Elastic Distributed Dask Application by executing this [instruction](https://us-south.git.cloud.ibm.com/ibmcws-application-instance-samples/dask-ego)
&nbsp;

2. Now your Elastic Distributed Dask Application is in "Started" state, copy following url from the UI console 'Outputs' section:
-  dask_scheduler_address
-  dask_scheduler_dashboard_address

&nbsp;
&nbsp;



![alt text](https://raw.githubusercontent.com/IBM/wmla-assets/master/WMLA-learning-journey/shared-images/Dask_1_image.png)

&nbsp;
&nbsp;

3. Start Dask Diagnostic Dashboard by copying 'dask_scheduler_dashboard_address' to any browser,  observe there is single dask cuda worker starting up ONLY.

![alt text](https://raw.githubusercontent.com/IBM/wmla-assets/master/WMLA-learning-journey/shared-images/Dask_2_image.png)

&nbsp;
&nbsp;



4. Add Jupyter notebook in Spectrum Conductor Instance Group -> https://www.ibm.com/support/knowledgecenter/SSZU2E_2.4.1/managing_notebooks/notebook_add.html?view=kc

&nbsp;
&nbsp;

5. Start Notebook in an Instance Group -> https://www.ibm.com/support/knowledgecenter/SSZU2E_2.4.1/managing_instances/notebooks_start.html


6. Download notebook [dask_xgb_training.ipynb] from repo and upload it to Conductor Jupyter server
![alt text](https://raw.githubusercontent.com/IBM/wmla-assets/master/WMLA-learning-journey/shared-images/Dask_3_image.png)

7.  Execute the notebook

