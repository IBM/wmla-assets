
## Summary
&nbsp;
&nbsp;
Elastic Distributed Inference EDI is a feature available for Watson Machine Learning Accelerator.  EDI is a solution aimed to manage production inference across a cluster.  EDI will enable inference requests to be managed not only distributed across the cluster, but also with dynamic or elastic resource consumption.  Being robust, secure, and scalable, EDI provides the tools to manage your machine learning models, track the demands for each inference, and optimize the usage of the hardware.  EDI simplifies your production deployment by automatically generating REST and gRPC APIs for inference for your model. The management APIs allow full configuration and new deployments of any inference model and give you control over limits for the model.  The inference APIs enable full scalability to the hardware and dynamic resource allocation based on current loads.  As the load of inference requests for a model increases, EDI can scale up additional instances of that model to ensure every transaction progresses smoothly, and thus derive insights from the trained models to drive key value for time to market.  As the load decreases, EDI will scale down to release unused resource and support other workloads such as Elastic Distributed Training (EDT) and Hyperparameter Optimization (HPO), and drive hardware resource utilization.


&nbsp;
&nbsp;



## Description
In this module we will use a [Jupyter notebook](https://github.com/IBM/wmla-assets/blob/master/WMLA-learning-journey/elastic-distributed-training-module/elastic_distributed_training_demonstration.ipynb) to walk through the process of taking an PyTorch model from the community,  making the needed code changes to optimize and running multiple inference services in a single GPU using Elastic Distributed Inference.     When you have completed this code pattern, you will understand how to:


&nbsp;
&nbsp;



## Instructions

The detailed steps for this tutorial can be found in the associated [Jupyter notebook](https://github.com/IBM/wmla-assets/blob/master/WMLA-learning-journey/elastic-distributed-training-module/elastic_distributed_training_demonstration.ipynb).  Learn how to:


