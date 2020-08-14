

In this tutorial we will be performing a basic computer vision image classification example using the Deep Learning Impact functionality within Watson Machine Learning Accelerator. This tutorial identifies whether the images contain clothes, dresses, clothes on a person, and dresses on a person. You could of course use whatever data you'd like in your example.

## Learning objectives

After completing this tutorial, you'll understand how to:

* Get a feel for the deep learning workflow
* Classify images with PowerAI Enterprise
* Build a model using PowerAI Enterprise
* Become more familiar with the IBM Power Systems server ecosystem

## Estimated time

* The end-to-end tutorial takes approx 3 hours, which includes about 50 minutes of model training, plus installation and configuration as well as driving model through the GUI.

## Prerequisites

The tutorial requires access to a GPU-accelerated IBM Power Systems server model AC922 or S822LC. In addition to acquiring a server, there are multiple options to access Power Systems servers listed on the [PowerAI Developer Portal](https://developer.ibm.com/linuxonpower/deep-learning-powerai/try-powerai/).

## Steps

### Step 1. Download, install and configure the IBM PowerAI Enterprise Evaluation

* Download the IBM PowerAI Enterprise Evaluation software from the [IBM software repository](https://epwt-www.mybluemix.net/software/support/trial/cst/programwebsite.wss?siteId=303&tabId=569&w=rc9ehqk&p=18tkuuj28%20). This is a 4.9 GB download and requires an IBMid.
* Install and configure PowerAI Enterprise using the instructions listed in the [IBM Knowledge Center](https://www.ibm.com/support/knowledgecenter/en/SSFHA8_1.1.1/powerai_evaluation.html) or the [OpenPOWER Power-Up User Guide](https://power-up.readthedocs.io/en/latest/Running-paie.html).

### Step 2. Download the insturmented VGG-19 model for TensorFlow

Download all of the files in the [https://git.ng.bluemix.net/ibmconductor-deep-learning-impact/dli-1.2.0-tensorflow-samples/tree/master/tensorflow-1.10/vgg19](https://git.ng.bluemix.net/ibmconductor-deep-learning-impact/dli-1.2.0-tensorflow-samples/tree/master/tensorflow-1.10/vgg19) directory.

### Step 3. Download the pre-trained weights

Use the following code to download the pre-trained weights from TensorFlow. More information can be found on their [GitHub repo](https://github.com/tensorflow/models/tree/master/research/slim).

```bash
mkdir <pretrained weight directory>
cd <pretrained weight directory>
wget http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz
tar –zxvf vgg_19_2016_08_28.tar.gz
```
Modify user access and group to ensure PowerAI Enterprise can read the weight file.
```bash
$ chown -R egoadmin:egoadmin <pretrained weight directory>
```

### Step 4. Download the data sets

For this tutorial, we're going to use a tool called [googliser](https://github.com/teracow/googliser), which searches Google Images. It is a simple shell script with no prerequisites. 

Use the following commands to run googliser and create four data sets in their own directories.

* `dresses_with_model`
* `dresses_without_model`
* `clothes_with_model`
* `clothes_without_model`

```bash

$ git clone https://github.com/teracow/googliser

$ cd googliser

$ ./googliser.sh --phrase "dresses with model" --title "dresses_with_model" --upper-size 200000 --lower-size 2000 --failures 0 -n 400 -N
 googliser.sh - 2018-07-26 PID:[43878]

 -> processing query: "dresses with model"
 -> searching Google:       10/10 result groups downloaded.      522 results!
 -> acquiring images:      400/400 downloaded and      115/     522 failed. (22%)

 -> All done!

$ ./googliser.sh --phrase "dresses only" --title "dresses_without_model" --upper-size 200000 --lower-size 2000 --failures 0 -n 400 -N
 googliser.sh - 2018-07-26 PID:[86968]

 -> processing query: "dresses only"
 -> searching Google:       10/10 result groups downloaded.      536 results!
 -> acquiring images:      400/400 downloaded and      122/     536 failed. (23%)

 -> All done!

$ ./googliser.sh --phrase "clothes with model" --title "clothes_with_model" --upper-size 200000 --lower-size 2000 --failures 0 -n 400 -N
 googliser.sh - 2018-07-26 PID:[14331]

 -> processing query: "clothes with model"
 -> searching Google:       10/10 result groups downloaded.      615 results!
 -> acquiring images:      400/400 downloaded and      194/     615 failed. (33%)

 -> All done!

$ ./googliser.sh --phrase "clothes only" --title "clothes_without_model" --upper-size 200000 --lower-size 2000 --failures 0 -n 400 -N
 googliser.sh - 2018-07-26 PID:[40210]

 -> processing query: "clothes only"
 -> searching Google:       10/10 result groups downloaded.      630 results!
 -> acquiring images:      400/400 downloaded and      112/     630 failed.  (34%)

 -> All done!
```

Create two directories, `images_without_model` and `images_with_model`, and move the images into those directories.

```
mkdir images_with_model
mv dresses_with_model/* images_with_model
mv clothes_with_model/* images_with_model

mkdir images_without_model
mv dress_without_model/* images_without_model
mv clothes_only/ images_without_model
```

Modify user access and group to ensure PowerAI Enterprise can read these files.

```bash
$ chown -R egoadmin:egoadmin images_with*
```

### Step 5. Load data into PowerAI Enterprise

Associate the images with PowerAI Enterprise by creating a new data set.

    ![](images/new-deep-learning-1.png)

1. In the **Datasets** tab, select **New**.

    ![](images/new-deep-learning-2.png)

1. Click **Images for Object Classification**. When presented with a dialog box, provide a unique name (for example, 'CodePatternDS') and select the folder that contains the images obtained in the previous step. The other fields are fine to use with the default settings. When you're ready, click **Create**.

With your data in PowerAI Enterprise, you can begin the next step, building a model.

### Step 6. Build the model

Modify the user access and group to ensure that PowerAI Enterprise can read vgg19 model files.

```bash
$ chown -R egoadmin:egoadmin vgg19model/
```

1. Select the **Models** tab and click **New**.

    ![](images/new-model-1.png)

1. Select **Add Location**.

    ![](images/ModelCreation1.png)

1. Select **TensorFlow** as the Framework.

    ![](images/ModelCreation2.png)

1. Select **TensorFlow-VGG19** for your new model, and click **Next**.

    ![](images/ModelCreation3.png)

1. Ensure that the training engine is set to `singlenode` and that the data set points to the one you just created.

    ![](images/new-model-3.png)

    **Note**: Set the Base learning rate to 0.001 because larger values might lead to exploding gradients.

    ![](images/new-model-4.png)

The model is now ready to be trained.

### Step 7. Run Training

1. Back at the **Models** tab, select **Train** to view the models you can train, then select the model you created in the previous step.

    ![](images/run-training-1.png)

1. Use the pretrained weight file you downloaded in the previous step by specifying the directory. Make sure that the files have a `.ckpt` extension. Click **Start Training**.

    ![](images/run-training-2.png)

### Step 8. Inspect the training run

1. From the **Train** submenu of the **Models** tab, select the model that is training by clicking the link.

    ![](images/inspect-1.png)

1. Navigate from the **Overview** panel to the **Training** panel, and click the most recent link. You can watch as the results roll in.

    ![](images/inspect-2.png)

### Step 9. Create an inference model

From the **Training** view, click **Create Inference Model**.

![](images/inference-1.png)

This creates a new model in the **Models** tab. You can view it by going to the **Inference** submenu.

![](images/inference-2.png)

### Step 10. Test it out

1. Go back to the **Models** tab, select the new inference model, and click **Test**. At the new Testing overview screen, select **New Test**.

    ![](images/testing-1.png)

1. Download inference test images into your local disk.
    * [Inference_images.zip](https://github.ibm.com/IBMCode/Code-Tutorials/tree/master/use-computer-vision-with-dli-on-powerai-enterprise/images/Inference_images.zip)
 
1. Unzip Inference_images.zip and use the *Browse* option to load 6 images. Click *Start Test*.

    ![](images/testing-2.png)

1. Wait for the test state to change from `RUNNING` to `FINISHED`.

    ![](images/testing-3.png)

1. Click the link to view the results of the test.

    ![](images/testing-4.png)

As you can see, the images are available as a thumbnail preview along with their classified label and probability.

![](images/testing-5.png)

## Summary

We hope that you have enjoyed reading this tutorial. Happy hacking and good luck on creating your next model with PowerAI Enterprise.
