
# Prerequisites
1. Same as listed in mnist_convnet sample

# Steps
1. Steps to get WMLA REST host, setting the WMLA_REST_HOST, WMLA commandline, logon to WMLA are the same as described in mnist_convnet sample so won't be repeated here.
1. You can see the modifications and comments:

        $ diff mnist_convnet.py  mnist_convnet.py.org
        13d12
        < import os
        18,29d16
        < print("--- WMLA environment variables start ---")
        < # DATA_DIR: a directory can be used to locate data for training; for example, /gpfs/mydatafs
        < DATA_DIR=os.environ.get("DATA_DIR", "")
        < print("DATA_DIR=", DATA_DIR)
        <
        < # RESULT_DIR: a directory can be used to store data for training such as models, checkpoints;
        < # for example, RESULT_DIR=/gpfs/myresultfs/{username}/batchworkdir/wmla-ns-41 where username
        < # is username used to logon to WMLA
        < RESULT_DIR=os.environ.get("RESULT_DIR", "")
        < print("RESULT_DIR=", RESULT_DIR)
        < print("--- WMLA environment variables end ---")
        <
        80,81c67
        < # Reduce epochs to reduce run time
        < epochs = 1
        ---
        > epochs = 15
        94,98d79
        <
        < # Need to store under model directory so we can retrieve later
        < model_dir = os.path.join(RESULT_DIR, "model")
        < os.mkdir(model_dir)
        < model.save(model_dir)


1. Submit the python script using command line. In the following example, we request to run on a CPU. Remove this option to request for GPU. The example also shows "wmla-ns-33" as id of the job. You will use this id to query status and so on.

        # ./dlicmd.py --exec-start "tensorflow" --rest-host $WMLA_REST_HOST --appName my-tf-app --workerDeviceType cpu  --numWorker 1 --model-main mnist_convnet.py --model-dir mnist_convnet

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
        --- WMLA environment variables start ---
        DATA_DIR= /gpfs/mydatafs
        RESULT_DIR= /gpfs/myresultfs/admin/batchworkdir/wmla-ns-33
        --- WMLA environment variables end ---
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
        422/422 [==============================] - 72s 171ms/step - loss: 0.3847 - accuracy: 0.8828 - val_loss: 0.0886 - val_accuracy: 0.9752
        Test loss: 0.0913003608584404
        Test accuracy: 0.972599983215332

1. Download the model saved by running:

        ./dlicmd.py --exec-trainresult wmla-ns-33 --rest-host $WMLA_REST_HOST
        Train result is saved to dlpd-model-6148355239355062-872481634wmla-ns-44-result.zip

# Self Test
Here's an example of running self-test script:

        $ ./self-test.sh  -r  wmla-console-wmla-ns.apps.spasms.cp.ibm.com -u admin -p password

        $ ./self-test.sh  -r  wmla-console-wmla-ns.apps.spasms.cp.ibm.com -u admin -p password


        INFO: download WMLA commandline
        INFO: cmd=wget https://wmla-console-wmla-ns.apps.spasms.cp.ibm.com/ui/tools/dlicmd.py -O dlicmd.py --no-check-certificate --quiet

        INFO: logon to WMLA
        INFO: cmd=./dlicmd.py --logon --rest-host wmla-console-wmla-ns.apps.spasms.cp.ibm.com --username admin --password password

        INFO: submit the python script
        INFO: cmd=./dlicmd.py --exec-start tensorflow --rest-host wmla-console-wmla-ns.apps.spasms.cp.ibm.com --appName my-tf-app --workerDeviceType cpu  --numWorker 1 --model-main mnist_convnet.py --model-dir ../mnist_convnet
        Copying files and directories ... Content size: 1.5K { "execId": "wmla-ns-1", "appId": "wmla-ns-1" }

        INFO: Get app Id
        INFO: appId=wmla-ns-1

        INFO: wait for app wmla-ns-1 to finish
        INFO: ./dlicmd.py --exec-get wmla-ns-1 --rest-host wmla-console-wmla-ns.apps.spasms.cp.ibm.com
        INFO: state=
        INFO: state=PENDING_CRD_SCHEDULER
        INFO: state=PENDING_CRD_SCHEDULER
        INFO: state=RUNNING
        ...
        INFO: state=RUNNING
        INFO: state=RUNNING
        INFO: state=RUNNING
        INFO: state=FINISHED

        INFO: get training stdout and stderr
        INFO: cmd=./dlicmd.py --exec-trainlogs wmla-ns-1 --rest-host wmla-console-wmla-ns.apps.spasms.cp.ibm.com
        INFO: expect trainoutlogs contains *Test loss:* and *Test accuracy:*. OK

        INFO: get training result
        INFO: cmd=./dlicmd.py --exec-trainresult wmla-ns-1 --rest-host wmla-console-wmla-ns.apps.spasms.cp.ibm.com
        INFO: expect trainresult contains *Train result is saved to dlpd-model*. OK
