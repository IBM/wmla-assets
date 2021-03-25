# TensorFlow image classification models sample

## About this sample
This sample enable following classification models integrated in tf.keras.applications
* vgg19
* inceptionv3
* mobilenetv2
* resnet50, resnet50v2, resnet101, resnet101v2, resnet152, resnet152v2
* densenet121, densenet169, densenet201

**Note:** This project is intended to be an example of how to use deep learning models. 
If you have any questions about this project, ask us on Slack: http://ibm.biz/joinslackcws.

### Supported deep learning frameworks:

* TensorFlow 2.x (Single node training)
* TensorFlow 2.x (Distributed training with TensorFlow) 

## Requirements to use this sample
- Ensure that IBM Spectrum Conductor Deep Learning Impact is installed and configured successfully.
- Ensure that the required frameworks are installed.
- Ensure that you have an available dataset. For example, you can use the flower dataset from TensorFlow. See the [TensorFlow website](https://www.tensorflow.org/tutorials/image_retraining) for more information or [download  directly](http://download.tensorflow.org/example_images/flower_photos.tgz).

## Prepare data set
All the models now only support to train with TensorFlow records with following format.

    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64)
    }

Create TensorFlow records data set in IBM Spectrum Conductor Deep Learning Impact. Do the following:

  1. From the cluster management console, navigate to **Workload > Spark > Deep Learning**.
  2. Select the **Dataset** tab and click **New**.
  3. Select **Images for Object Classification** option.
  4. Fill in the required dataset information and click **Create**.

## Train this sample model
To start a training run with the sample model, complete the following steps: 

1. Upload all the model files to the server.  
2. From the cluster management console, navigate to **Workload > Spark > Deep Learning**.
3. Navigate to the **Models** tab and click **New**.
4. Click **Add Location** and input the required fields, including: TensorFlow v2 as the framework and the upload location of the model files. 
5. Click **Next** and input the required fields for the model. **IMPORTANT**: Choose **tf.keras** as the API type.
6. Select the newly created model and click **Train** to start a training run.

## Start an inference job
To start an inference job with the trained sample model, complete the following steps: 

1. From the cluster management console, navigate to **Workload > Spark > Deep Learning**.
2. Navigate to the **Models** tab and click on the sample model that you trained. 
3. Select the **Training** tab, and select the finished training run that you want to use as an inference model and click **Create Inference Model**.
4. Go back to the **Deep Learning** page, select the inference model and click **Inference**.
5. Select the files that you want to predict and click **Start Inference**.
6. After the inference job is finished, click on the job to see prediction results.

## Publish an inference model as service

Before publishing an inference model as a service, make sure to obtain and install the elastic distributed inference technical preview package:

1.   From the cluster management console, navigate to **Workload > Deep Learning**.
2.   Navigate to the **Models** tab and click the inference model that you want to publish as a service.
3.   Click **Publish**.
4.   Set your values and click **Publish an inference model**. You can use the default values provided or add additional attributes, such as: img_height, img_width, img_depth, maximum_batchsize, num_classes or label_file.
5.   Go to the **Elastic Distributed Inference** tab, select the service that you just created and click **Start**. The inference service is now available for use. 
