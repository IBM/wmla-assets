
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
- Make your dataset available
- Set up API end point and log on
- Submit job via API
- Monitor running job
- Retrieve output and saved models
- Debug any issues


## Changes to your code

Note that the code sections below show a comparison between the "before" and "EDT enabled" versions of the code using `diff`.

1. Import the required additional libraries required for Elastic Distributed Training and set up environment variables. Note the additional EDT helper scripts: `edtcallback.py`, `emetrics.py` and `elog.py`. These need to be copied to the same directory as your modified code. Sample versions can be found in the tarball in the tutorial repo; additionally they can be downloaded from http://ibm.biz/WMLA-samples.

&nbsp;
&nbsp;
<!-- ![alt text](https://raw.githubusercontent.com/IBM/wmla-assets/master/WMLA-learning-journey/shared-images/1_model_update.png) -->

<pre style="font-size: 10px; color:white; background-color:black">
import os                                                                       import os
import copy                                                                     import copy

<span style="color:lime;">                                                                              &gt; # EDT changes - additional libraries</span>
<span style="color:lime;">                                                                              &gt;</span>
<span style="color:lime;">                                                                              &gt; import argparse</span>
<span style="color:lime;">                                                                              &gt; import sys</span>
<span style="color:lime;">                                                                              &gt; from os import environ</span>
<span style="color:lime;">                                                                              &gt; import json</span>
<span style="color:lime;">                                                                              &gt; import torch.nn.functional as F</span>
<span style="color:lime;">                                                                              &gt;</span>
<span style="color:lime;">                                                                              &gt; # EDT changes - setting up enviroment variables. Note the additional helper s</span>
<span style="color:lime;">                                                                              &gt; # for EDT: edtcallback.py, emetrics.py and elog.py. These need to sit in the </span>
<span style="color:lime;">                                                                              &gt; # as this code. Sample versions can be downloaded from http://ibm.biz/WMLA-sa</span>
<span style="color:lime;">                                                                              &gt;</span>
<span style="color:lime;">                                                                              &gt; path=os.path.join(os.getenv(&quot;FABRIC_HOME&quot;), &quot;libs&quot;, &quot;fabric.zip&quot;)</span>
<span style="color:lime;">                                                                              &gt; print(path)</span>
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
<span style="color:lime;">                                                                              &gt; model_path = os.environ[&quot;RESULT_DIR&quot;]+&quot;/model/saved_model&quot;</span>
<span style="color:lime;">                                                                              &gt; tb_directory = os.environ[&quot;LOG_DIR&quot;]+&quot;/tb&quot;</span>
<span style="color:lime;">                                                                              &gt; print (&quot;model_path: %s&quot; %model_path)</span>
<span style="color:lime;">                                                                              &gt; print (&quot;tb_directory: %s&quot; %tb_directory)</span>
<span style="color:lime;">                                                                              &gt;</span>
# Data augmentation and normalization for training                              # Data augmentation and normalization for training
# Just normalization for validation                                             # Just normalization for validation
</pre>
&nbsp;
&nbsp;


2.  Replace the data loading functions with EDT compatible functions that return a tuple containing two items of type `torch.utils.data.Dataset`:

&nbsp;
&nbsp;
<!-- ![alt text](https://raw.githubusercontent.com/IBM/wmla-assets/master/WMLA-learning-journey/shared-images/2_model_update.png)
![alt text](https://raw.githubusercontent.com/IBM/wmla-assets/master/WMLA-learning-journey/shared-images/3_model_update.png) -->

<pre style="font-size: 10px; color:white; background-color:black">
# Data augmentation and normalization for training                              # Data augmentation and normalization for training
# Just normalization for validation                                             # Just normalization for validation
<span style="color:red;">data_transforms = {                                                           &lt;</span>
<span style="color:red;">    'train': transforms.Compose([                                             &lt;</span>
<span style="color:red;">        transforms.RandomResizedCrop(224),                                    &lt;</span>
<span style="color:red;">        transforms.RandomHorizontalFlip(),                                    &lt;</span>
<span style="color:red;">        transforms.ToTensor(),                                                &lt;</span>
<span style="color:red;">        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])    &lt;</span>
<span style="color:red;">    ]),                                                                       &lt;</span>
<span style="color:red;">    'val': transforms.Compose([                                               &lt;</span>
<span style="color:red;">        transforms.Resize(256),                                               &lt;</span>
<span style="color:red;">        transforms.CenterCrop(224),                                           &lt;</span>
<span style="color:red;">        transforms.ToTensor(),                                                &lt;</span>
<span style="color:red;">        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])    &lt;</span>
<span style="color:red;">    ]),                                                                       &lt;</span>
<span style="color:red;">}                                                                             &lt;</span>

<span style="color:red;">data_dir = '/home/username/hymenoptera_data'                                  &lt;</span>
<span style="color:red;">image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),          &lt;</span>
<span style="color:red;">                                          data_transforms[x])                 &lt;</span>
<span style="color:red;">                  for x in ['train', 'val']}                                  &lt;</span>
<span style="color:red;">dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4 &lt;</span>
<span style="color:red;">                                             shuffle=True, num_workers=4)     &lt;</span>
<span style="color:red;">              for x in ['train', 'val']}                                      &lt;</span>
<span style="color:red;">dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}         &lt;</span>
<span style="color:red;">class_names = image_datasets['train'].classes                                 &lt;</span>

<span style="color:lime;">                                                                              &gt; # EDT changes - replace the data loading functions with ones that return a tu</span>
<span style="color:lime;">                                                                              &gt; # two items of type torch.utils.data.Dataset</span>
<span style="color:lime;">                                                                              &gt; def getDatasets():</span>
<span style="color:lime;">                                                                              &gt;     data_transforms = {</span>
<span style="color:lime;">                                                                              &gt;         'train': transforms.Compose([</span>
<span style="color:lime;">                                                                              &gt;             transforms.RandomResizedCrop(224),</span>
<span style="color:lime;">                                                                              &gt;             transforms.RandomHorizontalFlip(),</span>
<span style="color:lime;">                                                                              &gt;             transforms.ToTensor(),</span>
<span style="color:lime;">                                                                              &gt;             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]</span>
<span style="color:lime;">                                                                              &gt;         ]),</span>
<span style="color:lime;">                                                                              &gt;         'val': transforms.Compose([</span>
<span style="color:lime;">                                                                              &gt;             transforms.Resize(256),</span>
<span style="color:lime;">                                                                              &gt;             transforms.CenterCrop(224),</span>
<span style="color:lime;">                                                                              &gt;             transforms.ToTensor(),</span>
<span style="color:lime;">                                                                              &gt;             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]</span>
<span style="color:lime;">                                                                              &gt;         ]),</span>
<span style="color:lime;">                                                                              &gt;     }</span>
<span style="color:lime;">                                                                              &gt;     </span>
<span style="color:lime;">                                                                              &gt;     return (datasets.ImageFolder(os.path.join(dataDir, 'train'), data_transfo</span>
<span style="color:lime;">                                                                              &gt;             datasets.ImageFolder(os.path.join(dataDir, 'val'), data_transform</span>
</pre>


&nbsp;
&nbsp;

3.   Replace the training and testing loops with the EDT equivalent function. This requires the creation of a function `main`.
&nbsp;
You could also potentially specify parameters in the API call and pass these parameters into the model.
&nbsp;

- Extract parameters from the rest API call:
&nbsp;
&nbsp;
<!-- ![alt text](https://raw.githubusercontent.com/IBM/wmla-assets/master/WMLA-learning-journey/shared-images/4_model_update.png) -->

<pre style="font-size: 10px; color:white; background-color:black">

<span style="color:lime;">                                                                              &gt; # EDT changes - define main function and parse parameters for training</span>
<span style="color:lime;">                                                                              &gt;</span>
<span style="color:lime;">                                                                              &gt; def main():</span>
<span style="color:lime;">                                                                              &gt;     </span>
<span style="color:lime;">                                                                              &gt;     parser = argparse.ArgumentParser(description='PyTorch MNIST Example')</span>
<span style="color:lime;">                                                                              &gt;     parser.add_argument('--batchsize', type=int, default=64, metavar='N',</span>
<span style="color:lime;">                                                                              &gt;                         help='input batch size for training (default: 64)')</span>
<span style="color:lime;">                                                                              &gt;     parser.add_argument('--numWorker', type=int, default=100, metavar='N',</span>
<span style="color:lime;">                                                                              &gt;                         help='maxWorker')</span>
<span style="color:lime;">                                                                              &gt;     parser.add_argument('--epochs', type=int, default=5, metavar='N',</span>
<span style="color:lime;">                                                                              &gt;                         help='input epochs for training (default: 64)')</span>
<span style="color:lime;">                                                                              &gt;</span>
<span style="color:lime;">                                                                              &gt;     args, unknow = parser.parse_known_args()</span>
<span style="color:lime;">                                                                              &gt;  </span>
<span style="color:lime;">                                                                              &gt;     print('args: ', args)</span>
<span style="color:lime;">                                                                              &gt;     print('numWorker args:', args.numWorker) </span>
<span style="color:lime;">                                                                              &gt;     print('batch_size args:', args.batchsize)</span>
<span style="color:lime;">                                                                              &gt;     print('epochs args:', args.epochs)</span>
<span style="color:lime;">                                                                              &gt;</span>
</pre>

&nbsp;
&nbsp;
- Instantiate Elastic Distributed Training instance & launch EDT job with specific parameters:
  - epoch
  - effective_batch_size
  - max number of GPUs per EDT training job
  - checkpoint creation frequency in number of epochs
&nbsp;
&nbsp;
<!-- ![alt text](https://raw.githubusercontent.com/IBM/wmla-assets/master/WMLA-learning-journey/shared-images/5_model_update.png) -->

<pre style="font-size: 10px; color:white; background-color:black">

<span style="color:lime;">                                                                              &gt; # EDT changes - Replace the training and testing loops with EDT equivalents</span>
<span style="color:lime;">                                                                              &gt;</span>
<span style="color:lime;">                                                                              &gt; # model_conv = train_model(model_conv, criterion, optimizer_conv,</span>
<span style="color:lime;">                                                                              &gt; #                          exp_lr_scheduler, num_epochs=25)</span>

<span style="color:aqua;">model_conv = train_model(model_conv, criterion, optimizer_conv,               |     edt_m = FabricModel(model_conv, getDatasets, F.nll_loss, optimizer_conv, </span>
<span style="color:aqua;">                         exp_lr_scheduler, num_epochs=25)                     |     edt_m.train(args.epochs, args.batchsize, args.numWorker,checkpoint_freq=5</span>

<span style="color:lime;">                                                                              &gt; if __name__ == '__main__':</span>
<span style="color:lime;">                                                                              &gt;     print('sys.argv: ', sys.argv)</span>
<span style="color:lime;">                                                                              &gt;     main()</span>
</pre>

&nbsp;
&nbsp;
