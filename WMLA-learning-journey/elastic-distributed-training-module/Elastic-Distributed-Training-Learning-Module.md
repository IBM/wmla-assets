
## Summary
&nbsp;
&nbsp;
Watson Machine Learning Accelerator Elastic Distributed Training (EDT) simplifies the distribution of training workloads for the data scientist.   The model distribution is transparent to the end user, with no need to specifically know the topology of the distribution.   The usage is simple: define a maximum GPU count for training jobs and Watson Machine Learning Accelerator will schedule the jobs simultaneously on the existing cluster resources. GPU allocation for multiple jobs can grow and shrink dynamically, based on fair share or priority scheduling, without interruption to the running jobs.

&nbsp;
&nbsp;
EDT enables multiple Data Scientists to share GPUs in a dynamic fashion, increasing data scientists' productivity whilst also increasing overall GPU utilization.

&nbsp;
&nbsp;



## Description
This is the first module of the Watson ML Accelerator Learning Journey.  In this module we will use a [Jupyter notebook](https://github.com/IBM/wmla-assets/blob/master/WMLA-learning-journey/elastic-distributed-training-module/elastic_distributed_training_demonstration.ipynb) to walk through the process of taking an PyTorch model from the community,  making the needed code changes to distribute the training using Elastic Distributed Training.     When you have completed this code pattern, you will understand how to:

- Train the PyTorch Model with Elastic Distributed Training
- Monitor running job status and know how to debug any issues

&nbsp;
&nbsp;



## Instructions

The detailed steps for this tutorial can be found in the associated [Jupyter notebook](https://github.com/IBM/wmla-assets/blob/master/WMLA-learning-journey/elastic-distributed-training-module/elastic_distributed_training_demonstration.ipynb).  Learn how to:

- Make changes to your code
- Make dataset available
- Set up API end point and log on
- Submit job via API
- Monitor running job
- Retrieve output and saved models
- Debug any issues


## Changes to your code

Note that the code sections below show a comparison between the "before" and "EDT enabled" versions of the code using `diff`.


1.  Import the required Elastic Distributed Training libraries and environment variables:

&nbsp;
&nbsp;
<!-- ![alt text](https://raw.githubusercontent.com/IBM/wmla-assets/master/WMLA-learning-journey/shared-images/1_model_update.png) -->

<pre style="font-size: 10px; color:white; background-color:black">
import torch                                                                    import torch
import torch.nn as nn                                                           import torch.nn as nn
<span style="color:lime;">                                                                              &gt; import torch.nn.functional as F</span>
import torch.optim as optim                                                     import torch.optim as optim
<span style="color:red;">from torch.optim import lr_scheduler                                          &lt;</span>
<span style="color:red;">import numpy as np                                                            &lt;</span>
import torchvision                                                              import torchvision
<span style="color:aqua;">from torchvision import datasets, models, transforms                          | from torchvision import datasets, transforms</span>
<span style="color:aqua;">import matplotlib.pyplot as plt                                               | from torch.optim import lr_scheduler</span>
<span style="color:aqua;">import time                                                                   | from pathlib import Path</span>
<span style="color:lime;">                                                                              &gt;</span>
<span style="color:lime;">                                                                              &gt; import sys</span>
import os                                                                       import os
<span style="color:aqua;">import copy                                                                   | from os import environ</span>
<span style="color:lime;">                                                                              &gt; import json</span>


<span style="color:aqua;"># Data augmentation and normalization for training                            | #  Importing libraries and setting up enviroment variables </span>
<span style="color:aqua;"># Just normalization for validation                                           | path=os.path.join(os.getenv(&quot;FABRIC_HOME&quot;), &quot;libs&quot;, &quot;fabric.zip&quot;)</span>
<span style="color:aqua;">data_transforms = {                                                           | print(path)</span>
<span style="color:lime;">                                                                              &gt; sys.path.insert(0,path)</span>
<span style="color:lime;">                                                                              &gt; from fabric_model import FabricModel</span>
<span style="color:lime;">                                                                              &gt; from edtcallback import EDTLoggerCallback</span>
<span style="color:lime;">                                                                              &gt;</span>
<span style="color:lime;">                                                                              &gt; dataDir = environ.get(&quot;DATA_DIR&quot;)</span>
<span style="color:lime;">                                                                              &gt; if dataDir is not None:</span>
<span style="color:lime;">                                                                              &gt;     print(&quot;dataDir is: %s&quot;%dataDir)</span>
<span style="color:lime;">                                                                              &gt; else:</span>
<span style="color:lime;">                                                                              &gt;     print(&quot;Warning: not found DATA_DIR from os env!&quot;)</span>
<span style="color:lime;">                                                                              &gt;</span>
<span style="color:lime;">                                                                              &gt;</span>
<span style="color:lime;">                                                                              &gt; model_path = os.environ[&quot;RESULT_DIR&quot;]+&quot;/model/saved_model&quot;</span>
<span style="color:lime;">                                                                              &gt; tb_directory = os.environ[&quot;LOG_DIR&quot;]+&quot;/tb&quot;</span>
<span style="color:lime;">                                                                              &gt; print (&quot;model_path: %s&quot; %model_path)</span>
<span style="color:lime;">                                                                              &gt; print (&quot;tb_directory: %s&quot; %tb_directory)</span>
<span style="color:lime;">                                                                              &gt;</span>
</pre>
&nbsp;
&nbsp;


2.  Replace the data loading functions with ones that are compatible with EDT - the data loading function must return a tuple containing two items of type `torch.utils.data.Dataset`:

&nbsp;
&nbsp;
<!-- ![alt text](https://raw.githubusercontent.com/IBM/wmla-assets/master/WMLA-learning-journey/shared-images/2_model_update.png)
![alt text](https://raw.githubusercontent.com/IBM/wmla-assets/master/WMLA-learning-journey/shared-images/3_model_update.png) -->

<pre style="font-size: 10px; color:white; background-color:black">
<span style="color:lime;">                                                                              &gt; # Data Loading function for EDT</span>
<span style="color:lime;">                                                                              &gt;</span>
<span style="color:lime;">                                                                              &gt; def getDatasets():</span>
<span style="color:lime;">                                                                              &gt;     data_transforms = {</span>
    'train': transforms.Compose([                                                   'train': transforms.Compose([
        transforms.RandomResizedCrop(224),                                              transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),                                              transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),                                                          transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),                                                                             ]),
    'val': transforms.Compose([                                                     'val': transforms.Compose([
        transforms.Resize(256),                                                         transforms.Resize(256),
        transforms.CenterCrop(224),                                                     transforms.CenterCrop(224),
        transforms.ToTensor(),                                                          transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),                                                                             ]),
<span style="color:aqua;">}                                                                             |     }</span>
<span style="color:red;">                                                                              &lt;</span>
<span style="color:red;">data_dir = 'data/hymenoptera_data'                                            &lt;</span>
<span style="color:red;">image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),          &lt;</span>
<span style="color:red;">                                          data_transforms[x])                 &lt;</span>
<span style="color:red;">                                                                              &lt;</span>
<span style="color:red;">                                                                              &lt;</span>
<span style="color:red;">                  --- SNIP ---                                                &lt;</span>
<span style="color:red;">                                                                              &lt;</span>
<span style="color:red;">                                                                              &lt;</span>
<span style="color:red;">    time_elapsed = time.time() - since                                        &lt;</span>
<span style="color:red;">    print('Training complete in {:.0f}m {:.0f}s'.format(                      &lt;</span>
<span style="color:red;">        time_elapsed // 60, time_elapsed % 60))                               &lt;</span>
<span style="color:red;">    print('Best val Acc: {:4f}'.format(best_acc))                             &lt;</span>

<span style="color:aqua;">    # load best model weights                                                 |     return (datasets.ImageFolder(os.path.join(dataDir, 'train'), data_transfo</span>
<span style="color:aqua;">    model.load_state_dict(best_model_wts)                                     |             datasets.ImageFolder(os.path.join(dataDir, 'val'), data_transform</span>
<span style="color:red;">    return model                                                              &lt;</span>

</pre>


&nbsp;
&nbsp;

3.   Replace the training and testing loops with EDT’s train function.
&nbsp;
You could also potentially specify parameters in the Rest API call and pass these parameters into the model.
&nbsp;

- extract parameters from rest API call
&nbsp;
&nbsp;
<!-- ![alt text](https://raw.githubusercontent.com/IBM/wmla-assets/master/WMLA-learning-journey/shared-images/4_model_update.png) -->

<pre style="font-size: 10px; color:white; background-color:black">
<span style="color:lime;">                                                                              &gt;     # Extract parameters for training</span>
<span style="color:lime;">                                                                              &gt;     parser = argparse.ArgumentParser(description='PyTorch MNIST Example')</span>
<span style="color:lime;">                                                                              &gt;     parser.add_argument('--batchsize', type=int, default=64, metavar='N',</span>
<span style="color:lime;">                                                                              &gt;                         help='input batch size for training (default: 64)')</span>
<span style="color:lime;">                                                                              &gt;     parser.add_argument('--numWorker', type=int, default=100, metavar='N',</span>
<span style="color:lime;">                                                                              &gt;                         help='maxWorker')</span>
<span style="color:lime;">                                                                              &gt;     parser.add_argument('--epochs', type=int, default=5, metavar='N',</span>
<span style="color:lime;">                                                                              &gt;                         help='input epochs for training (default: 64)')</span>
<span style="color:lime;">                                                                              &gt;     args, unknow = parser.parse_known_args()</span>
<span style="color:lime;">                                                                              &gt;     print('args: ', args)</span>
<span style="color:lime;">                                                                              &gt;     print('numWorker args:', args.numWorker) </span>
<span style="color:lime;">                                                                              &gt;     print('batch_size args:', args.batchsize)</span>
<span style="color:lime;">                                                                              &gt;     print('epochs args:', args.epochs)</span>
<span style="color:lime;">                                                                              &gt;     </span>
</pre>

&nbsp;
&nbsp;
- Instantiate Elastic Distributed Training instance & launch EDT job with specific parameters:
  - epoch
  - effective_batch_size
  - max number of GPU per EDT training
  - checkpoint creation frequency in number of epochs
&nbsp;
&nbsp;
<!-- ![alt text](https://raw.githubusercontent.com/IBM/wmla-assets/master/WMLA-learning-journey/shared-images/5_model_update.png) -->

<pre style="font-size: 10px; color:white; background-color:black">
<span style="color:lime;">                                                                              &gt;     # Replace the training and testing loops with EDT equivalents</span>
<span style="color:lime;">                                                                              &gt;     edt_m = FabricModel(model_conv, getDatasets, F.nll_loss, optimizer_conv, </span>
<span style="color:lime;">                                                                              &gt;     edt_m.train(args.epochs, args.batchsize, args.numWorker,checkpoint_freq=5</span>

</pre>

&nbsp;
&nbsp;
