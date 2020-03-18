
## Summary
&nbsp;
&nbsp;
Watson Machine Learning Accelerator Elastic Distributed Training (EDT) simplifies the distribution of training workloads for the data scientist.   The model distribution is transparent to the end user, with no need to specifically know the topology of the distribution.   The usage is simple: define a maximum GPU count for training jobs and Watson Machine Learning Accelerator will schedule the jobs simultaneously on the existing cluster resources. GPU allocation for multiple jobs can grow and shrink dynamically, based on fair share or priority scheduling, without interruption to the running jobs.

&nbsp;
&nbsp;
EDT enables multiple Data Scientists to share GPUs in a dynamic fashion, increasing data scientists' productivity whilst also increasing overall GPU utilization.

&nbsp;
&nbsp;



## Description
This is the first module of the Watson ML Accelerator Learning Journey.  In this module we will use a [Jupyter notebook](https://github.com/IBM/wmla-assets/blob/master/WMLA-learning-journey/elastic-distributed-training-module/elastic_distributed_training_demonstration.ipynb) to walk through the process of taking an PyTorch model from the community,  making the needed code changes to distribute the training using Elastic Distributed Training.     When you have completed this code pattern, you will understand how to:

- Train the PyTorch Model with Elastic Distributed Training
- Monitor running job status and know how to debug any issues

&nbsp;
&nbsp;



## Instructions

The detailed steps for this tutorial can be found in the associated [Jupyter notebook](https://github.com/IBM/wmla-assets/blob/master/WMLA-learning-journey/elastic-distributed-training-module/elastic_distributed_training_demonstration.ipynb).  Learn how to:

- Make changes to your code
- Make your dataset available
- Set up API end point and log on
- Submit job via API
- Monitor running job
- Retrieve output and saved models
- Debug any issues


## Changes to your code

Note that the code sections below show a comparison between the "before" and "EDT enabled" versions of the code using `diff`.

1. Import the required additional libraries required for Elastic Distributed Training and set up environment variables. Note the additional EDT helper scripts: `edtcallback.py`, `emetrics.py` and `elog.py`. These need to be copied to the same directory as your modified code. Sample versions can be found in the tarball in the tutorial repo; additionally they can be downloaded from http://ibm.biz/WMLA-samples.

&nbsp;
&nbsp;
<!-- ![alt text](https://raw.githubusercontent.com/IBM/wmla-assets/master/WMLA-learning-journey/shared-images/1_model_update.png) -->
![image1](https://raw.githubusercontent.com/IBM/wmla-assets/master/WMLA-learning-journey/elastic-distributed-training-module/shared-images/screenshot1.png)
&nbsp;
&nbsp;


2.  Replace the data loading functions with EDT compatible functions that return a tuple containing two items of type `torch.utils.data.Dataset`:

&nbsp;
&nbsp;
<!-- ![alt text](https://raw.githubusercontent.com/IBM/wmla-assets/master/WMLA-learning-journey/shared-images/2_model_update.png)
![alt text](https://raw.githubusercontent.com/IBM/wmla-assets/master/WMLA-learning-journey/shared-images/3_model_update.png) -->
![image2](https://raw.githubusercontent.com/IBM/wmla-assets/master/WMLA-learning-journey/elastic-distributed-training-module/shared-images/screenshot2.png)



&nbsp;
&nbsp;

3.   Replace the training and testing loops with the EDT equivalent function. This requires the creation of a function `main`.
&nbsp;
You could also potentially specify parameters in the API call and pass these parameters into the model.
&nbsp;

- Extract parameters from the rest API call:
&nbsp;
&nbsp;
<!-- ![alt text](https://raw.githubusercontent.com/IBM/wmla-assets/master/WMLA-learning-journey/shared-images/4_model_update.png) -->
![image3](https://raw.githubusercontent.com/IBM/wmla-assets/master/WMLA-learning-journey/elastic-distributed-training-module/shared-images/screenshot3.png)


&nbsp;
&nbsp;
- Instantiate Elastic Distributed Training instance & launch EDT job with specific parameters:
  - epoch
  - effective_batch_size
  - max number of GPUs per EDT training job
  - checkpoint creation frequency in number of epochs
&nbsp;
&nbsp;
<!-- ![alt text](https://raw.githubusercontent.com/IBM/wmla-assets/master/WMLA-learning-journey/shared-images/5_model_update.png) -->
![image4](https://raw.githubusercontent.com/IBM/wmla-assets/master/WMLA-learning-journey/elastic-distributed-training-module/shared-images/screenshot4.png)


&nbsp;
&nbsp;
