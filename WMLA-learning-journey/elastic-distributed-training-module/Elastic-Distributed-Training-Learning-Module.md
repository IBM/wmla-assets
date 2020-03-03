## Following format from CodePattern: https://developer.ibm.com/patterns/locate-and-count-items-with-object-detection/



## Summary
&nbsp;
&nbsp;
Watson Machine Learning Accelerator Elastic Distributed Training (EDT) simplifies the distribution logic for the data scientist.   Te model distribution is transparent to the end user, without coding the topology of the distribution.   The usage is simple: Define a maximum GPU count for training jobs and Watson Machine Learning Accelerator will schedule the jobs simultaneously on the existing cluster resources. GPU allocation for multiple jobs can grow and shrink dynamically, based on fair share or priority scheduling, without interruption. 

&nbsp;
&nbsp;
EDT enables multiple Data Scientists sharing GPUs in dynamic fashion, driving data scientists productivity while driving overall GPU utilization. 

&nbsp;
&nbsp;



## Description
This is the first module of Watson ML Learning Journery.  In this module we wil use a [Jupyter Notebook] (https://github.com/IBM/wmla-assets/blob/master/WMLA-learning-journey/elastic-distributed-training-module/elastic_distributed_training_demonstration.ipynb) to walk through the process of taking an PyTorch model from community,  making the code changes to distribute the training using Elastic Distributed Training.     When you have completed this code pattern, you will understand how to:

- Train the PyTorch Model with Elastic Distributed Training
- Monitor running job status and how to debug any issues

&nbsp;
&nbsp;



## Instructions

Find the detailed steps for this pattern in the Jupyter Notebook.  Learn how to:

    - Changes to your code
    - Making dataset available
    - Set up API end point and log on
    - Submit job via API
    - Monitor running job
    - Retrieve output and saved models
        Output
        Saved Models
    - Debugging any issues


## Changes to your code

1.  Import Elastic Distributed Training library and environment variable

&nbsp;
&nbsp;
![alt text](https://github.com/IBM/wmla-assets/blob/master/WMLA-learning-journey/shared-images/1_model_update.png)
&nbsp;
&nbsp;

2.  Replace the data loading functions with ones that are compatible with EDT - the data loading function must return a tuple containing two items of type torch.utils.data.Dataset

&nbsp;
&nbsp;
![alt text](https://github.com/IBM/wmla-assets/blob/master/WMLA-learning-journey/shared-images/2_model_update.png)
![alt text](https://github.com/IBM/wmla-assets/blob/master/WMLA-learning-journey/shared-images/3_model_update.png)
&nbsp;
&nbsp;

3.   Replace the training and testing loops with EDT’s train function
&nbsp;
You could specify parameters in Rest API and pass these parameters into the model.
&nbsp;

- extract parameters from rest API call 
&nbsp;
&nbsp;
![alt text](https://github.com/IBM/wmla-assets/blob/master/WMLA-learning-journey/shared-images/4_model_update.png)
&nbsp;
&nbsp;
- Instantiate Elastic Distributed Training instance & Launch EDT job with specific parameters 
  - epoch
  - effective_batch_size
  - max number of GPU per EDT training
  - checkpt creation per number of EPOCH
&nbsp;
&nbsp;
![alt text](https://github.com/IBM/wmla-assets/blob/master/WMLA-learning-journey/shared-images/5_model_update.png)
&nbsp;
&nbsp;

