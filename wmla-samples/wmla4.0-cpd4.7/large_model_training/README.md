
## Large Model Training with IBM Watson Machine Learning Accelerator

[DeepSpeed](https://www.deepspeed.ai/) and [PyTorch FullyShardedDataParallel (FSDP)](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) are two of the distributed training libraries you can leverage to train large models that can not fit into single device. In this tutorial, we will show you how to leverage `DeepSpeed` and `FSDP` in Watson Machine Learning Accelerator for distributed training.

Steps overview:
1. [Requirements](#req)
2. [Extending the Watson Machine Learning Accelerator runtime image](#runtime)
3. [Running large model training with Watson Machine Learning Accelerator](#run)

### 1. Requirement <a name="req"></a>
- This tutorial is avaiable in Watson Machine Learning Accelerator from Cloud Pak for Data Version 4.7.2.

### 2. Extending the Watson Machine Learning Accelerator runtime image <a name="runtime"></a>
Watson Machine Learning Accelerator provides a default runtime image the includes libraries like TensorFlow and PyTorch to run different workloads. Watson Machine Learning Accelerator can also run with a customized runtime image that will install other necessary packages for your workloads.

#### 2.1 Build and publish the image

Build and publish a new runtime image that will include FSDP and DeepSpeed.
`FSDP` is available in the released default runtime image of Watson Machine Learning Accelerator, but `DeepSpeed` will need be installed.

Below is the [Dockerfile](./Dockfile) to build new runtime image based on Watson Machine Learning Accelerator default runtime image. You can change the base image to the one deployed in your cluster. Besides the Python packages, we also installed `gcc-c++`and added the CUDA library path into `LIBRARY_PATH`, which is required by `DeepSpeed` to compile Ops just-in-time (JIT) at runtime. Besides `DeepSpeed`, you can also install `transformers` and other necessary packages, but please note packages like `transformers` and `datasets` may have issues to run in FIPS enabled environment.

```
From cp.icr.io/cp/cpd/wml-accelerator-runtime:v4.2.0-rt23.1-py3.10

ENV LIBRARY_PATH=/opt/anaconda3/envs/dlipy3/lib:$LIBRARY_PATH

RUN source activate dlipy3 && \
    microdnf -y install gcc-c++ && \
    pip install deepspeed==0.9.5 transformers==4.30.2 datasets==2.13.1 accelerate==0.20.3
```

Use the command below to build the image and publish the image to the registry where the Watson Machine Learning Accelerator training pod will later pull from.
```
podman build -f Dockerfile -t <your-registry>/<your-image>
podman push <your-registry>/<your-image>
```
Note: specify the registry and the name of the new runtime image. Make sure to update to an image registry that is accessible from the cluster.

#### 2.2 Set up an image pull secret
To enable the training pod to be able to pull the new image, you need to [add access to the image registry](https://docs.openshift.com/container-platform/4.12/openshift_images/managing_images/using-image-pull-secrets.html) if it is not configured yet.

#### 2.3 Use the new runtime image for training
There are two options to use customized runtime image. The first way is to change the global default runtime image as described in [Step 5](https://www.ibm.com/docs/en/wmla/4.0?topic=configuration-create-custom-runtime) ensuing that all the training workloads will use the new runtime image. The second way is to specify the runtime image for each training workload submission. In this tutorial, we will use the second way.

### 3. Running large model training with Watson Machine Learning Accelerator Rest API <a name="run"></a>

Training can be submitted to Watson Machine Learning Accelerator through this [RESTful API](https://www.ibm.com/docs/en/wmla/4.0?topic=reference-deep-learning-rest-api#execsPost).

Below is a sample `data` payload to submit training:
```
{
    "args": "--exec-start distPyTorch \
             --msd-image-name <your-registry>/<your-image> \
             --msd-env HF_HOME=/tmp --msd-env TORCH_EXTENSIONS_DIR=/tmp \
             --user-cmd -- bash bash_torchrun.sh train.py --deepspeed ds.config",
    "hardwareSpec": {
        "nodes" : {
            "num_nodes": 4,
            "mem": {
                "size": "16g"
            },
            "gpu": {
                "num_gpu": 1
            },
            "cpu": {
                "units": "0.25"
            }
        }
    }
}
```

In the formdata `data` payload, you can define training command and resource requirement.
1. `args` defines the training runtime and commands.
- `--msd-image-name` is defining the runtime image that will be used for the training workload.
- `--msd-env` can be used to define additional environment variables that will be exposed in the training workload. `TORCH_EXTENSIONS_DIR` environment variable can be added because `DeepSpeed` will compile required Ops and save it into this directory, because it is not allowed to write into the default directory in the training pod. For the same reason, `HF_HOME` can be added if you are training with `transformer`.
- From `--user-cmd`, we define the training script and its supported arguments. [bash_torchrun.sh](./bash_torchrun.sh) is the script to launch training workload with [torchrun](https://pytorch.org/docs/stable/elastic/run.html). After that, you can append your training script (`train.py` for example) and arguments for that script (`--deepspeed ds.config` for example).

2. `hardwareSpec` defines the hardware resource that are used in the training workload.
- `num_nodes` defines the number of training pods and `--nnodes=` in `torchrun` will use it.
- `num_gpu` defines the number of GPUs that are assigned to each training pod. This value will be used by `--nproc-per-node` in `torchrun`.
- You can also define other resources like `cpu` and `memory` for each training pod.

For the formdata `file`, go the directory where training script `train.py` locates and package the training codes as below.
```
tar -cvf /tmp/llm_train.modelDir.tar .
```

Use the following command to submit the training workload. Refer to the IBM Cloud Pak for Data Platform API documentation on [Authentication](https://cloud.ibm.com/apidocs/cloud-pak-data/cloud-pak-data-4.7.0#using-authorization-bearer-token) to get the bearer token.

```
curl -k -X POST -H "Authorization: Bearer <TOKEN>" -H "Accept: application/json" -F data='{"args": "", "hardwareSpec": {}}' -F file=@/tmp/llm_train.modelDir.tar "https://<wmla-console-route>/platform/rest/deeplearning/v1/execs"
```
