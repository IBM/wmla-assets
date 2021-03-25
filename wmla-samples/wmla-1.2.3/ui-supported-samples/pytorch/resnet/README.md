# resnet PyTorch sample

## About this sample
This sample is configured to work with IBM Spectrum Conductor Deep Learning Impact 1.2.5 and is enabled to work with all the deep learning features included with IBM Spectrum Conductor Deep Learning Impact, such as training, inference, elastic distributed training, deep learning insights, hyperparameter tuning and more. This sample covers:
- Requirements to use this sample
- Train this sample model
- Start an inference job

The resnet sample model is a modified version of [resnet](https://pytorch.org/). This sample model can be used with the cifar10 dataset. For details about model definitions, parameters and training procedure, refer to [cuda-convnet](https://code.google.com/p/cuda-convnet/). To learn more about cifar10, refer to the [readme file](https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/README.md) in the cifar10 project.

**Note:** This project is intended to be an example of how to use deep learning models. 
If you have any questions about this project, ask us on Slack: http://ibm.biz/joinslackcws.


### Supported deep learning frameworks:

* PyTorch 1.x (Single node training)
* PyTorch 1.x (Elastic distributed training)

### Supported Resnet models:
* ResNet-18
* ResNet-34
* ResNet-50
* ResNet-101
* ResNet-152


## Requirements to use this sample
- Ensure that IBM Spectrum Conductor Deep Learning Impact is installed and configured successfully.
- Ensure that the required frameworks are installed.
- Ensure that you have an available dataset. For this sample model, it is recommended that you use the cifar10 dataset which has 32x32 image size and 10 classes. If you use the cifar10 dataset, no additional changes need to be made to the cifar10 dataset.


### Before you begin:
1. Download the [CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar.html).
2. Once downloaded, prepare the dataset by following these steps:
  1. Untar the downloaded dataset. This will yield a folder called "cifar-10-batches-py".
  2. Create two new directories: one called "train", and the other called "test".
  3. Copy the cifar-10-batches-py directory to both the train and test folders created in the previous step.
3. Import the dataset to be used with the model. Do the following:
  1. From the cluster management console, navigate to **Workload > Deep Learning**.
  2. Select the **Dataset** tab and click **New**.
  3. Select the **Any** option.
  4. Select **COPY**  in the "type" drop down menu. In the "training folder" text box, input the full path to the "train" directory created in a previous step. Likewise, in the "testing folder" text box, input the full path to the "test" directory.
  4. Select the dataset name and the Spark instance group for this dataset and click **Create**.
  

## Train this sample model
To start a training run with the sample model, complete the following steps: 

1. Upload all the model files to the server.  
2. From the cluster management console, navigate to **Workload > Deep Learning**.
3. Navigate to the **Models** tab and click **New**.
4. Click **Add Location** and input the required fields, including: PyTorch as the framework and the upload location of the model files. 
5. Click **Next** and input the required fields for the model. **IMPORTANT**: All training engines are available with this model so you can select any of the training engine options.
6. Select the newly created model and click **Train** to start a training run.


## Start an inference job
To start an inference job with the trained sample model, complete the following steps: 

1. From the cluster management console, navigate to **Workload > Deep Learning**.
2. Navigate to the **Models** tab and click on the sample model that you trained. 
3. Select the **Training** tab, and select the finished training run that you want to use as an inference model and click **Create Inference Model**.
4. Go back to the **Deep Learning** page, select the inference model and click **Inference**.
5. Select the files that you want to predict and click **Start Inference**.
6. After the inference job is finished, click on the job to see prediction results.


## Limitation
* This sample model is only for image classification.


## Files included with this sample

| File             | Description        |
| ---------------- | :----------------: |
| README.md        | README file        |
| License.txt      | License file       |
| main.py          | Model file for training       |
| elastic-main.py  | Model file for Elastic distributed training   |
| inference.py     | Model file for inference         |
| monitor.py       | Model logging configuration file **Note**: The default model name is **resnet18** in this file, when use other resnet model, update this name accordingly       |
| ps.conf          | Model configuration file        |
