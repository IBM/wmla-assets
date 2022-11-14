# Guide To Samples
The samples are listed from simplest to more involved. If this is the first time for you, we suggest to start with **mnist_convnet**; otherwise, just pick the ones with titles and descriptions closest to what you are trying to do. For each sample, there is a README.md file with the details how to run the sample, as well as a **self-test.sh** which autotmates the instructions.

## mnist_convnet
Start with this sample to get a feel how submit DL jobs to WMLA using command line, monitor job status, get job output file. You will download publicly available Tensorflow code and submit without any modification.

## mnist_convnet_storage
This sample extends the mnist_convnet example by showing:
-  How to use **storage** provided by WMLA to **save models and retrieve models** after the training is done.
- How to submit code in a directory in the case where there are more than one files.

We made minor modifications to the downloaded Tensorflow code. The original code is saved in .org extension. In addition, the number of epochs is reduced to reduce the training time.

## tf2x_mnist
This sample shows how to run a training job with **multiple workers and multiple devices** with **Tensorflow distributed MultiWorkerMirroredStrategy** (multi-node multi-device synchronous training). In addition, it also shows the following:
  - emetrics used by Watson Machine Learning (WML)
  - passing parameters to the python code
