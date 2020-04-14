

## Summary

One of the first steps in developing a new AI model is to define the characterization of the training cycle. The variables which describe these elements of the model configuration are called hyperparameters.

For the same model, hyperparameters may be varied based on the training data set to achieve the optimal training result.

Hyperparameter optimization in Watson Machine Learning Accelerator (WML-A) utilizes a number of algorithms to automatically optimize hyperparameters, including: Random Search, Tree-structured Parzen Estimator Approach (TPE), Hyperband and Bayesian Optimization based on Gaussian Process, or bringing in their own algorithm,  prior to the commencement of the training process.

WML-A automates this process by sampling from the training data set and instantiating multiple small training jobs across cluster resources (sometimes tens, sometimes thousands of jobs depending on the complexity of the model) and then selects the most promising combination of hyperparameters to return to the data scientist. To ensure the best possible outcomes, the hyperparameter selection process uses the four aforementioned separate search algorithms to accurately select and then reflect the best combination of variables.

Afterwards, data scientists can kick off training with the optimal values of hyperparameters found during this automated tuning.

Data scientists also have the option to select the hyperparameter search type (one out of four) along with a defined range of values for each hyperparameter to test out during the tuning phase. This gives expert data scientists the starting points for WML-A to work with and explore from based on their own expertise.



## Description
In this module we will use multiple notebooks to walk through the process of taking the an PyTorch model from the community, making the needed code changes and identifying optimal hyper-parameter with Watson ML Accelerator Hyperparameter Optimization, including:

-  Introduction of Watson ML Accelerator Hyperparameter Optimization -> https://github.com/IBM/wmla-assets/blob/master/HPO-demonstration/WMLA-HPO-state.ipynb
-  Define custom search space -> https://github.com/IBM/wmla-assets/blob/master/HPO-demonstration/WMLA-HPO-Custom-Experiment.ipynb
-  Bring Your Own Algorithm -> https://github.com/IBM/wmla-assets/blob/master/HPO-demonstration/WMLA-HPO-Plugin-Search-Algorithm.ipynb
-  Distribute Hyperparameter Optimization tasks with Elastic Distributed Training -> https://github.com/IBM/wmla-assets/blob/master/HPO-demonstration/HPO-EDT/WMLA-HPO-Hyperband-EDT-external.ipynb


## Instructions

The detailed steps for this tutorial can be found in the associated xx.  
Learn how to:
- Make changes to your code
- Set up API end point and log on
- Submit job via API
- Monitor running job
- Retrieve output and saved models
- Debug any issues


## Changes to your code
TODO
