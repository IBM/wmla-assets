## Following format from CodePattern: https://developer.ibm.com/patterns/locate-and-count-items-with-object-detection/



## Summary
Watson Machine Learning Accelerator Elastic Distributed Training (EDT) transforms static monolithic training into a dynamic process that is resilient to failures and automatically scales GPU allocation while training.   The model distribution is transparent to the end user, without coding the topology of the distribution.    EDT dynamically allocates GPUs across multiple running job without any job termination.   EDT enables multiple Data Scientists sharing GPUs in dynamic fashion, while driving overall GPU utilization.





Elastic Distributed Training enables multiple data scientists sharing GPU resources in elastic manner.    


In this tutorial, you will run multiple model trainings simultaneously with the Watson Machine Learning Accelerator Elastic Distributed Training feature (EDT). With EDT, we can distribute model training across multiple GPUs and compute nodes. The distribution of training jobs is elastic, which means GPUs are dynamically allocated: GPUs can be added or removed while executing and without having to kill the job. Since the scheduler dynamically allocates the GPUs, you don’t need to code the GPU topology into the model (for example, GPU01 out of node 1 and GPU01 out of node 2). You can take a model built on a stand-alone system, EDT does the distribution, and it’s transparent to the end user.





- Elastic training 
	- DLI handles the resource topology and allocations in real time, open source requires rigid allocation set before execution
	- Sharing scenarios by dynamically changing, adding/removing, resources
	- Especially in scenarios like this: with open source, if a job will consume all GPUs for 8 hrs and you have a 20 min job to execute you need to wait for 8 hr before you can begin. DLI will allocate resource immediately allowing the short running job to start and finish while the longer job continues to execute.

are designed for large scale distributed deep learning workloads. It transforms static monolithic training into a dynamic process that is resilient to failures and automatically scales GPU allocation while training.





## Description


## Flow


## Instruction
