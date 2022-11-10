This is an example of getting a Tensorflow python script from a public github, submit the script as-is to run as a WML-Accelerator (WMLA) job using WMLA command line, get the job status, and the training log. This is the **simplest sample** to get started with WML-Accelerator - without having to provide python deep learning scripts!

# Prerequisites
1. A running WMLA installation
1. A user name (e.g. admin) and password that you can log into WMLA
1. A client node where you can run OCP command 'oc' and wget. 'oc' command is only needed to get WMLA REST host. You can also get WMLA REST host from your cluster admin.

# Steps
1. Get the WMLA REST host. You can get this from running the following. In the example below, **wmla-ns** is the namespace in which WMLA was installed:

        # WMLA_REST_HOST=`oc get route wmla-console -n wmla-ns -o yaml -o jsonpath='{.spec.host}'`
        # echo $WMLA_REST_HOST
        wmla-console-wmla-ns.apps.spasms.ibm.com

1. Download the WMLA commandline:


        # wget $WMLA_REST_HOST/ui/tools/dlicmd.py -O dlicmd.py --no-check-certificate --quiet

1. Download the public Tensorflow python script:


        # wget https://github.com/keras-team/keras-io/raw/master/examples/vision/mnist_convnet.py -O mnist_convnet.py 

1. Logon to WMLA. In the following example, we use admin/password as username/password:


        # ./dlicmd.py --logon --rest-host $WMLA_REST_HOST --username admin --password password

1. Submit the python script using command line. In the following example, we request to run on a CPU. Remove this option to request for GPU. The example also shows "wmla-ns-33" as id of the job. You will use this id to query status and so on.

        # ./dlicmd.py --exec-start "tensorflow" --rest-host $WMLA_REST_HOST --appName my-tf-app --workerDeviceType cpu  --numWorker 1 --model-main mnist_convnet.py

        Copying files and directories ...
        Content size: 1.9K
        {
        "execId": "wmla-ns-33",
        "appId": "wmla-ns-33"
        }


1. Get status of the job. Here's an example:

        #./dlicmd.py --exec-get wmla-ns-33 --rest-host $WMLA_REST_HOST | grep state"
         "state": "RUNNING",


1. Once the state is in "FINISHED", get the training log:

        #./dlicmd.py --exec-trainoutlogs wmla-ns-33 --rest-host $WMLA_REST_HOST
        Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
        11490434/11490434 [==============================] - 1s 0us/step
        x_train shape: (60000, 28, 28, 1)
        60000 train samples
        10000 test samples
        Model: "sequential"
        _________________________________________________________________
        Layer (type)                Output Shape              Param #
        =================================================================
        conv2d (Conv2D)             (None, 26, 26, 32)        320

        max_pooling2d (MaxPooling2D  (None, 13, 13, 32)       0
        )

        conv2d_1 (Conv2D)           (None, 11, 11, 64)        18496

        max_pooling2d_1 (MaxPooling  (None, 5, 5, 64)         0
        2D)

        flatten (Flatten)           (None, 1600)              0

        dropout (Dropout)           (None, 1600)              0

        dense (Dense)               (None, 10)                16010

        =================================================================
        Total params: 34,826
        Trainable params: 34,826
        Non-trainable params: 0
        _________________________________________________________________
        Epoch 1/15
        422/422 [==============================] - 77s 181ms/step - loss: 0.3611 - accuracy: 0.8903 - val_loss: 0.0867 - val_accuracy: 0.9765
        Epoch 2/15
        422/422 [==============================] - 74s 176ms/step - loss: 0.1162 - accuracy: 0.9651 - val_loss: 0.0617 - val_accuracy: 0.9827
        Epoch 3/15
        422/422 [==============================] - 76s 180ms/step - loss: 0.0875 - accuracy: 0.9724 - val_loss: 0.0513 - val_accuracy: 0.9853
        Epoch 4/15
        422/422 [==============================] - 76s 180ms/step - loss: 0.0739 - accuracy: 0.9774 - val_loss: 0.0425 - val_accuracy: 0.9883
        Epoch 5/15
        422/422 [==============================] - 74s 174ms/step - loss: 0.0639 - accuracy: 0.9802 - val_loss: 0.0393 - val_accuracy: 0.9883
        Epoch 6/15
        422/422 [==============================] - 73s 174ms/step - loss: 0.0569 - accuracy: 0.9822 - val_loss: 0.0368 - val_accuracy: 0.9905
        Epoch 7/15
        422/422 [==============================] - 73s 173ms/step - loss: 0.0543 - accuracy: 0.9830 - val_loss: 0.0350 - val_accuracy: 0.9908
        Epoch 8/15
        422/422 [==============================] - 75s 177ms/step - loss: 0.0491 - accuracy: 0.9845 - val_loss: 0.0353 - val_accuracy: 0.9903
        Epoch 9/15
        422/422 [==============================] - 74s 176ms/step - loss: 0.0457 - accuracy: 0.9859 - val_loss: 0.0341 - val_accuracy: 0.9908
        Epoch 10/15
        422/422 [==============================] - 74s 176ms/step - loss: 0.0434 - accuracy: 0.9862 - val_loss: 0.0320 - val_accuracy: 0.9912
        Epoch 11/15
        422/422 [==============================] - 73s 174ms/step - loss: 0.0401 - accuracy: 0.9870 - val_loss: 0.0290 - val_accuracy: 0.9913
        Epoch 12/15
        422/422 [==============================] - 74s 175ms/step - loss: 0.0398 - accuracy: 0.9866 - val_loss: 0.0295 - val_accuracy: 0.9927
        Epoch 13/15
        422/422 [==============================] - 74s 175ms/step - loss: 0.0357 - accuracy: 0.9886 - val_loss: 0.0324 - val_accuracy: 0.9903
        Epoch 14/15
        422/422 [==============================] - 73s 172ms/step - loss: 0.0354 - accuracy: 0.9889 - val_loss: 0.0298 - val_accuracy: 0.9917
        Epoch 15/15
        422/422 [==============================] - 73s 173ms/step - loss: 0.0324 - accuracy: 0.9899 - val_loss: 0.0301 - val_accuracy: 0.9918
        Test loss: 0.02489200234413147
        Test accuracy: 0.9919999837875366

# Self Test
Here's an example of running self-test script:

        $ ./self-test.sh  -r  wmla-console-wmla-ns.apps.spasms.cp.ibm.com -u admin -p password

        INFO: download WMLA commandline
        INFO: cmd=wget https://wmla-console-wmla-ns.apps.spasms.cp.ibm.com/ui/tools/dlicmd.py -O dlicmd.py --no-check-certificate --quiet

        INFO: logon to WMLA
        INFO: cmd=./dlicmd.py --logon --rest-host wmla-console-wmla-ns.apps.spasms.cp.ibm.com --username admin --password password

        INFO: download the public Tensorflow python script
        INFO: cmd=wget https://github.com/keras-team/keras-io/raw/master/examples/vision/mnist_convnet.py -O mnist_convnet.py --quiet

        INFO: submit the python script
        INFO: cmd=./dlicmd.py --exec-start tensorflow --rest-host wmla-console-wmla-ns.apps.spasms.cp.ibm.com --appName my-tf-app --workerDeviceType cpu  --numWorker 1 --model-main mnist_convnet.py
        Copying files and directories ... Content size: 1.9K { "execId": "wmla-ns-57", "appId": "wmla-ns-57" }

        INFO: Get app Id
        INFO: appId=wmla-ns-57

        INFO: wait for app wmla-ns-57 to finish
        INFO: ./dlicmd.py --exec-get wmla-ns-57 --rest-host wmla-console-wmla-ns.apps.spasms.cp.fyre.ibm.com
        INFO: state=PENDING_CRD_SCHEDULER
        INFO: state=RUNNING
        ...
        INFO: state=RUNNING
        INFO: state=RUNNING
        INFO: state=FINISHED

        INFO: get training stderr and stdout
        INFO: cmd=./dlicmd.py --exec-trainlogs wmla-ns-57 --rest-host wmla-console-wmla-ns.apps.spasms.cp.ibm.com
        INFO: expect trainoutlogs contains *Test loss:* and *Test accuracy:*. OK

