# Watson Deep Learning Experiment UI for submitting distributed training

## Summary

This document describes validating connectivity between IBM Watson Studio Local, IBM Watson Machine Learning and IBM Watson Machine Learning Accelerator. 

Watson Machine Learning Accelerator Elastic Distributed Training enables multiple data scientists to share GPUs dynamically. This increases productivity and overall GPU utilization. In this tutorial, you will see how two data scientists submit Elastic Distributed Training requests using CP4D Deep Learning Experiment UI, accelerate their model training with WMLA Elastic Distributed Training, and dynamically share GPUs across two running training jobs.

## Objective

The objective of this tutorial is to teach the user to use Deep Learning Experiment in Cloud Pak for Data for more efficient training by running multiple runs and how IBM's Deep Learning evenly distributes the GPU for parallel job processes. In this example, we use a Pytorch MNIST model that analyzes handwriting in pictures and predict what the handwritten text says. Two training sessions are created at the same time. To create and run the experiment, you must have access to the following:

a. A data set to use for training and testing the model. This tutorial uses an MNIST data set for analyzing handwriting samples.

b. A training definition that contains model building code and metadata about how to run the experiment. For information on coding a training definition file, see [Coding guidelines for deep learning programs](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/ml_dlaas_code_guidelines.html).

c. A training execution command. The execution command must reference the Python code, pass the names of the training files, and can optionally specify metrics. The tutorial includes these components and instructions for downloading them and adding them to your experiment.

## Environments

- IBM Watson Studio Local on Cloud Pak for Data 
- IBM Watson Machine Learning on Cloud Pak for Data
- IBM Spectrum Conductor 
- IBM Watson Machine Learning Accelerator

- Pre-Requisites: Before completing the CP4D/WSL/WML/WMLA validation steps, please ensure connectivity is configured correctly according to the document Connecting IBM Watson Studio Local, IBM Watson Machine Learning and IBM Watson Machine Learning Accelerator.

## Submit two Deep Learning experiments

### Before you begin

1. Log into your WMLA cluster in the terminal.

1. Download the MNIST dataset and the corresponding PyTorch model and place them in the `DLI_DATA_FS` directory as follows:

    a. Please download both the `pytorch-mnist.zip` and `pytorch-mnist-edt-model.zip` from the following github page:
    https://github.com/IBM/wmla-learning-path/tree/master/datasets.

    b. Unzip the `pytorch-mnist.zip` dataset into the `DLI_DATA_FS` directory for IBM Spectrum Conductor Deep Learning Impact. You should see the following after you run the bash commands: 

    ```
    [root@supp20 pytorch-mnist]# pwd
    /mnt/dli_data_fs/pyrtoch-mnist
    [root@supp20 pytorch-mnist]# ls
    MNIST  processed  raw
    ```

    c. Create a path to the `pytorch-mnist` file.

1. Ensure the instance group(s) are running.

    a. Login to the IBM Spectrum Computing Cluster Management Console as the wml-user.

    b. Navigate to the instance group window.

    c. Start any instance groups that are not in *Ready* state by clicking the checkbox and selecting **Start**.

    <img src="https://raw.githubusercontent.com/IBM/wmla-learning-path/master/shared-images/cpd_wmla1.png">

### Create a new project

1. Login to the IBM Cloud Pak for Data Console as the wml-user.

<img src="https://raw.githubusercontent.com/IBM/wmla-learning-path/master/shared-images/01-unauth.png">
<img src="https://raw.githubusercontent.com/IBM/wmla-learning-path/master/shared-images/02-login.png">

1. Select the hamburger menu in the upper left corner, choose **Projects > New project**.

1. Select *Analytics project* when given the list of project types, and then select **Create an empty project**.

<img src="https://raw.githubusercontent.com/IBM/wmla-learning-path/master/shared-images/03-create-project.png">

1. Enter a project *Name* and select **Create** in lower right corner.

<img src="https://raw.githubusercontent.com/IBM/wmla-learning-path/master/shared-images/04-create-project-3.png">

### Create and run a Deep Learning experiment

In our scenario, the first data scientist submits their first Deep Learning job.

1. Select **Add to project** in the upper right corner.

<img src="https://raw.githubusercontent.com/IBM/wmla-learning-path/master/shared-images/05-assets-page.png">

1. Select Deep learning experiment.

1. Enter a *Name*, the path to the source file folder name (in our case, it is `pytorch-mnist`).

<img src="https://raw.githubusercontent.com/IBM/wmla-learning-path/master/shared-images/08-new-experiment.png">

1. Click **Add training definition** on the right hand side of the dialog. Go to the **New training definition** tab.

<img src="https://raw.githubusercontent.com/IBM/wmla-learning-path/master/shared-images/9-new-training-def.png">

1. In the **Add training definition** GUI, enter a *Name*, select **Browse** and choose the file `pytorch-mnist-edt-model.zip` (Gotten from here: https://github.com/IBM/wmla-learning-path/tree/master/datasets) for the **Training source code**.

1. Choose the **Framework**, in our example it is `pytorch-onnx 1.1-py 3.6`.

1. Enter the name of the Python script (included in `pytorch-mnist-edt-model.zip`), in our example it is `pytorch_mnist_EDT.py`.

1. Now the **Attributes** fields should be visible to the right. Choose the **Compute configuration** with 1 GPU.

1. For the **Distributed training type** select *Elastic distributed training*. For the purpose of this demonstration, choose 4 nodes.

1. Click **Create** in the lower right hand corner. You have created a new training definition with Elastic Distribution Training (EDT).

1. Select **Create and run** in the lower left hand of the new Deep Learning Experiment dialog. This is your first job.

Now the second data scientist also submits a job when the first is not finished its run.

1. To create your second job: 

    a. While the first job runs, repeat steps 1-3.

    b. When you click **Add training definition**, click the **Existing training definition** and choose the training definition that you have created.

    c. Repeat steps 8-11.

With both jobs that are running, the dynamic GPU allocation will occur automatically. Since there are four nodes, the EDT automatically distributes them evenly among the two jobs.

To view the results, go to IBM Spectrum Computing Cluster Management Console. Go to **Workload > Instance groups**, then click on the **Applications** tab.

Wait for a few minutes and you should see that the GPU slots for each of the two jobs are 2, such as this screenshot:

<img src="https://raw.githubusercontent.com/IBM/wmla-learning-path/master/shared-images/cpd_wmla4.png">

1. Select your Project and monitor the run and when complete will look similar to the following.

<img src="https://raw.githubusercontent.com/IBM/wmla-learning-path/master/shared-images/cpd_wmla2.png">

1. Select the **Completed Training Run** and you see something like this:

<img src="https://raw.githubusercontent.com/IBM/wmla-learning-path/master/shared-images/cpd_wmla3.png">
​
