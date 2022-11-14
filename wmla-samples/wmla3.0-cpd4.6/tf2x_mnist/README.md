
# Prerequisites
1. Same as listed in mnist_convnet sample

# Steps
1. Steps to get WMLA REST host, setting the WMLA_REST_HOST, WMLA commandline, logon to WMLA are the same as described in mnist_convnet sample so won't be repeated here.


1. Submit the python script using command line. In the following example, we request to run on a CPU. Remove this option to request for GPU. The example also shows "wmla-ns-33" as id of the job. You will use this id to query status and so on.

        # ./dlicmd.py --exec-start disttensorflow --rest-host $WMLA_REST_HOST --appName my-tf-app --workerDeviceType cpu  --numWorker 3  --model-main main.py --model-dir tf2x_mnist -- --epochs 10 --no-cuda true

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
        
        ...
        TF_CONFIG is founded
        {"cluster": {"worker": ["10.254.15.111:8889", "10.254.15.113:8889", "10.254.15.112:8889"]}, "task": {"index": 0, "type": "worker"}}
        Let's use 3 workers. is_chief = True
        Epoch 1/10
        157/157 [==============================] - ETA: 0s - loss: 2.2619 - accuracy: 0.1822
        Epoch 00001: saving model to /gpfs/myresultfs/admin/batchworkdir/wmla-ns-7/checkpoint/cp-0001.ckpt
        157/157 [==============================] - 28s 144ms/step - loss: 2.2619 - accuracy: 0.1822
        ...
        
        156/157 [============================>.] - ETA: 0s - loss: 1.0225 - accuracy: 0.7924
        Epoch 00010: saving model to /gpfs/myresultfs/admin/batchworkdir/wmla-ns-7/checkpoint/cp-0010.ckpt
        157/157 [==============================] - 15s 92ms/step - loss: 1.0224 - accuracy: 0.7924
        Train finished. Time cost: 2.63 minutes
        Test finished. Time cost: 0.06 minutes. Test loss: 0.943907, Test accuracy: 0.810500
        Model saved in path: /gpfs/myresultfs/admin/batchworkdir/wmla-ns-7/model

1. Download the model saved by running:

        ./dlicmd.py --exec-trainresult wmla-ns-33 --rest-host $WMLA_REST_HOST
        Train result is saved to dlpd-model-6148355239355062-872481634wmla-ns-44-result.zip

# Self Test
Here's an example of running self-test script:

        $ ./self-test.sh  -r  wmla-console-wmla-ns.apps.spasms.cp.ibm.com -u admin -p password

        
        INFO: download WMLA commandline
        INFO: cmd=wget https://wmla-console-wmla-ns.apps.spasms.cp.ibm.com/ui/tools/dlicmd.py -O dlicmd.py --no-check-certificate --quiet

        INFO: logon to WMLA
        INFO: cmd=./dlicmd.py --logon --rest-host wmla-console-wmla-ns.apps.spasms.cp.ibm.com --username admin --password password

        INFO: submit the python script
        INFO: cmd=./dlicmd.py --exec-start disttensorflow --rest-host wmla-console-wmla-ns.apps.spasms.cp.ibm.com --appName my-tf-app --numWorker 3 --workerDeviceType cpu --model-main main.py --model-dir ../tf2x_mnist -- --epochs 10 --no-cuda true
        Copying files and directories ... Content size: 4.8K { "execId": "wmla-ns-7", "appId": "wmla-ns-7" }

        INFO: Get app Id
        INFO: appId=wmla-ns-7

        INFO: wait for app wmla-ns-7 to finish
        INFO: ./dlicmd.py --exec-get wmla-ns-7 --rest-host wmla-console-wmla-ns.apps.spasms.cp.ibm.com
        INFO: state=PENDING_CRD_SCHEDULER
        INFO: state=RUNNING
        INFO: state=RUNNING
        INFO: state=RUNNING
        ...

        INFO: get training stdout and stderr
        INFO: cmd=./dlicmd.py --exec-trainlogs wmla-ns-7 --rest-host wmla-console-wmla-ns.apps.spasms.cp.ibm.com
        INFO: expect trainoutlogs contains *Test loss:* and *Test accuracy:*. OK
        INFO: expect trainoutlogs contains *TF_CONFIG is founded* . OK

        INFO: get training result
        INFO: cmd=./dlicmd.py --exec-trainresult wmla-ns-7 --rest-host wmla-console-wmla-ns.apps.spasms.cp.ibm.com
        INFO: expect trainresult contains *Train result is saved to dlpd-model*. OK
