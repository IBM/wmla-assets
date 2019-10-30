# Setting up WML-A base use cases
![IBM logo](images/image002.png)

### Introduction

This guide contains the setup that Lab Services performs at customers after a standard WML-A installation with regards to creating consumers, anaconda environments, notebooks, instance groups. 

This guide also contains the tests that Lab Services performs in order to declare that the environment is working properly. This includes submitting a spark job, running a notebook, running a DLI workload.

## Step 1: Configure the resource groups

1.	Log on as the cluster Admin user

2.	Open up the Resource Group configuration:

<img src="images/image003.png" width="50%">


3.	Select the ComputeHosts resource group:

<img src="images/image004.png" width="50%">


4.	Properly configure the number of slots to a value that makes sense. If the server is an 8-thread capable system, use 7 * number of processors. If it’s a 4-thread capable system, go with 3 * number of processors:

<img src="images/image005.png" width="50%">

5.	Optional, but recommended, change the resource selection method to static, and then select only the servers which will provide computing power (processor power) to the cluster:

<img src="images/image006.png" width="50%">

6.	Click Apply to commit the changes.

7.	Create a new resource group:

<img src="images/image007.png" width="50%">

8.	Call it GPUHosts:

<img src="images/image008.png" width="50%">

9.	The number of slots should use the advanced formula and equals the number of GPUs on the systems by using the keywork ngpus:

<img src="images/image009.png" width="50%">

10.	Optionally, but recommended, change the resource selection method to static and select the nodes which are GPU-capable:

<img src="images/image010.png" width="50%">

<img src="images/image011.png" width="50%">

11.	Under the “Members Host” column, click on “preferences” and select the attribute “ngpus” to be displayed:

<img src="images/image012.png" width="50%">

12.	Click on “Apply” and validate that the “Members Host” column now displays ngpus:

<img src="images/image013.png" width="50%">

13.	Finish the creation of the resource group by clicking on “Create”

14.	Go to Resources -> Resource Planning (slot) -> Resource Plan:

<img src="images/image014.png" width="50%">

15.	Change the allocation policy of the “ComputeHosts” resource group to balanced:

<img src="images/image015.png" width="50%">


## Step 2: Configure the roles

1.	To start with, we create a role of a Chief Data Scientist. The reason for such is so that we create a role with intermediate privileges between an Admin account and a Data Scientist account. This Chief Data scientist role has the authority of a data scientist plus additional privileges to start/stop instance groups. The idea is that users do not need to go up to a cluster Admin in order to start/stop their instance groups, instead they have the Chief Data Scientist do so.

2.	Go to Systems & Services -> Users -> Roles:

<img src="images/image016.png" width="50%">

3.	Select the “Data Scientist” role and duplicate it by clicking the duplicate button:

<img src="images/image017.png" width="50%">

4.	Call the new role “Chief Data Scientist”:

<img src="images/image018.png" width="50%">

5.	Select the “Chief Data Scientist” role and add a couple privileges:

 *	Conductor -> Spark Instance Groups -> Control

 *	Ego Services -> Services -> Control (exemplified below)

 *	Consumers and Resource Plans  -> Resource Plans -> View

<img src="images/image019.png" width="50%">

6.	Click Apply to commit the changes.


## Step 3: Configure the Consumer

1.	At the OS level, as root, on all nodes, create an OS group and user for the OS execution user: 
 *	groupadd demoexec
 *	useradd -g demoexec -m demoexec

2.	The GID and UID of the created user / group MUST be the same on all nodes.

3.	Now go to Resources -> Consumers

<img src="images/image020.png" width="50%">

4.	Click on “create a consumer”:

<img src="images/image021.png" width="50%">

5.	Name your consumer “DemoConsumer” (for best practices, use starting capital letters), and use “demoexec” in the list of users:

<img src="images/image022.png" width="50%">

6.	Further scroll down and input “demoexec” as the OS user for execution, and select the Management, Compute and GPU resource groups:

<img src="images/image023.png" width="50%">

7.	Click create to save.

8.	On the left side column, click on the “DemoConsumer” consumer you just created, and then click on “Create a consumer”:

<img src="images/image024.png" width="50%">

9.	Name your consumer “Anaconda3-DemoConsumer” (for best practices, use starting capital letters). Leave the “Inherit the user list and group list from parent consumer” selected:

<img src="images/image025.png" width="50%">

10.	Further scroll down and use “demoexec” as the operating system user for workload execution, and make sure all resource groups are selected:

<img src="images/image026.png" width="50%">

11.	Your “Anaconda3-DemoConsumer” should now appear as a child of “DemoConsumer”.


## Step 4: Create a user

1.	Go to “Systems & Services -> Users -> Accounts” 

<img src="images/image027.png" width="50%">

2.	Click on “Create New user account”:

<img src="images/image028.png" width="50%">

3.	Create a demonstration account called “DemoUser”:

<img src="images/image029.png" width="50%">

4.	Go to “Systems & Services -> Users -> Roles”:

<img src="images/image030.png" width="50%">

5.	Select your newly defined user (make sure you do NOT unselect Admin in the process) and then assign it to the “DemoConsumer” consumer you created in step 2:

<img src="images/image031.png" width="50%">

6.	Click OK and then Apply to commit the changes. Do not forget to click on Apply!!! 


## Step 4: Create an instance group for Spark workloads

1.	On EVERY cluster node, create the instance groups deployment directory structure. All commands are run as root:
*	mkdir -p /cwslocal/demoexec/
*	chown egoadmin:egoadmin /cwslocal/
*	chown demoexec:demoexec /cwslocal/demoexec/

2.	Go to “Workload -> Spark -> Spark Instance groups” 

<img src="images/image032.png" width="50%">

3.	Click on “Create a Spark Instance Group” to create your first instance group:

<img src="images/image033.png" width="50%">

4.	Name your instance group “Spark-DemoConsumer” (as a best  practice, use capital starting letters), choose “/cwslocal/demoexec/spark-democonsumer” (as a best practice, use all lowercase) as the deployment directory, “demoexec” as the OS execution user, and the latest available spark version:

<img src="images/image034.png" width="50%">

5.	Scroll down and click on the default consumer name that Conductor would create for you:

<img src="images/image035.png" width="50%">

6.	Click on the “X” to  delete that default consumer:

<img src="images/image036.png" width="50%">

7.	Select the “DemoConsumer” consumer and create a child consumer with the same consumer name as the one you just deleted on the previous step:

<img src="images/image037.png" width="50%">

8.	Click on “Create”, then on “Select”. Your consumer should now list something similar to what you see here:

<img src="images/image038.png" width="50%">

9.	Scroll down and select the “GPUHosts” resource group for the “Spark executors (GPU slots)”. Do not change any other configuration there.

<img src="images/image039.png" width="50%">

10.	Click on Create and Deploy Instance group. 

11.	Click on Continue to Instance Group

12.	Watch as your instance group gets deployed

<img src="images/image040.png" width="50%">


## Step 5: Import an Anaconda installer and create an anaconda environment

1.	Download The following file to your workstation:
*	https://repo.continuum.io/archive/Anaconda3-2019.03-Linux-ppc64le.sh

2.	Go to “Workload -> Spark -> Anaconda Management”.

<img src="images/image041.png" width="50%">

3.	Click on “Add”:

<img src="images/image042.png" width="50%">

4.	Fill out the information required:
*	Distribution name: Anaconda3
*	Select the anaconda file you downloaded at step 1 and upload it here
*	Anaconda version: 2019.03
*	Python version: 3
*	Operating system: Linux on Power 64-bit little endian (LE)

<img src="images/image043.png" width="50%">

5.	Click on “Add”.

6.	On all nodes, create a directory for an anaconda deployment for the proper execution user:
* mkdir -p /cwslocal/demoexec/anaconda
*	chown demoexec:demoexec /cwslocal/demoexec/anaconda

7.	Now select the distribution you just created and click on “Deploy”:

<img src="images/image044.png" width="50%">

8.	Fill in the information required:
*	Instance name: Anaconda3-DemoConsumer-PowerAI
*	Deployment directory: /cwslocal/demoexec/anaconda
*	Consumer: Anaconda3-DemoConsumer (which you created on step 2)
*	Resource group: compute hosts
*	Execution user: demoexec

<img src="images/image045.png" width="50%">

9.	Click on “Environment Variables”:

<img src="images/image046.png" width="50%">

10.	Click on “Add variable” and add variable “PATH” with the contents of “$PATH:/usr/bin”. This is mandatory due to bug #7649. Also, add a variable called “IBM_POWERAI_LICENSE_ACCEPT” with the contents of “yes”:

<img src="images/image047.png" width="50%">

11.	Click on “Deploy”. Watch as your anaconda environment gets deployed.

<img src="images/image047.png" width="50%">

12.	Create a powerai161.yml file on your workstation with the following content (notice the tabulation in the file!):

<img src="images/image048.png" width="50%">

13.	Now click on “Add” to add a conda environment:

<img src="images/image049.png" width="50%">

14.	Create a new environment from the powerai16.yml file you created, then click “Add”:

<img src="images/image050.png" width="50%">

15.	Watch as your environment gets created.


## Step 6: Create a notebook environment

1.	We leverage the IBM Spectrum Conductor-provided notebook. You can see it in “Workload -> Spark -> Notebook Management”.

<img src="images/image051.png" width="50%">

2.	Notice that there is a notebook called Jupyter, version 5.4.0. If you select it and click on “Configure” you can view the settings for this notebook:

<img src="images/image052.png" width="50%">

3.	The settings show properties such as:
*	the notebook package name
*	the scripts in use
*	Use (or not) of SSL
*	Anaconda required      (make sure this setting is selected!)

<img src="images/image053.png" width="50%">

4.	At the moment, due to a change on how Anaconda 2019-03 works, we need to apply a patch to the standard Jupyter 5.4.0 notebook’s deploy.sh script. This patched notebook can be found in: 

*	https://ibm.box.com/s/ps486rawe9o8sy21cyn2uxcv41sbhrql

*	Download this notebook to your workstation and replace the one that comes with Conductor by clicking on the “Browse” button and selecting the patched notebook:

<img src="images/image054.png" width="50%">

5.	Click on the “Update Notebook” button.


## Step 7: Create an instance group for notebook use

1.	On either node, create the data directory for the execution user within the shared filesystem:
* mkdir -p /cwsshare/demoexec/
* chown -R demoexec:demoexec /cwsshare/demoexec/

2.	Go to “Workload -> Spark -> Spark Instance Groups”:

<img src="images/image055.png" width="50%">

3.	Click on “New”:

<img src="images/image056.png" width="50%">

4.	Fill in the information with the following values:
*	Instance group name: Notebook-DemoConsumer
*	Deployment directory: /cwslocal/demoexec/notebook-democonsumer
*	Spark version: use the latest one available

<img src="images/image057.png" width="50%">

5.	Select the Jupyter 5.4.0 notebook and set the following properties:
*	data directory to: /cwsshare/demoexec/notebook-democonsumer
*	select the anaconda environment you created in Step 5 of this guide

<img src="images/image058.png" width="50%">

6.	Scroll down and click on the standard consumer which the process creates, we need to change it:

<img src="images/image059.png" width="50%">

7.	Scroll down until you find the standard suggested consumer name and click on the “X” to delete it:

<img src="images/image060.png" width="50%">

8.	Look for the “DemoConsumer” consumer, select it and create a child named “Notebook-DemoConsumer”. Click on “Create” and then on “Select”:

<img src="images/image061.png" width="50%">

9.	Your consumer should now look like something such as:

<img src="images/image062.png" width="50%">

10.	Scroll down and select the “GPUHosts” resource group for “Spark Executors (GPU slots)”. Do not change anything else.

<img src="images/image063.png" width="50%">

11.	Create on “Create and Deploy Instance Group” at the bottom of the page.

12.	Watch as your instance group gets deployed.

<img src="images/image064.png" width="50%">

13.	Once the instance group is deployed, start it by clicking on the “Start” button:

<img src="images/image065.png" width="50%">

14.	Once started, click on the “Notebook” tab and then on “Create notebook for users”:

<img src="images/image066.png" width="50%">

15.	Select the users you want to create a notebook for and click on “Create”:

<img src="images/image067.png" width="50%">

16.	Your notebooks should show up as Started after a while


## Step 8: Create an instance group for Deep Learning Impact with Elastic Distributed Search (EDT)

1.	Go to “Workload -> Spark -> Spark Instance Groups”:

<img src="images/image055.png" width="50%">

2.	Click on “New”:

<img src="images/image068.png" width="50%">

3.	Click on “Templates”:

<img src="images/image69.png" width="50%">

4.	Select “Use” for the dli-sig-template-2-2-0 template:

<img src="images/image070.png" width="50%">

5.	Fill in the following information:
*	Instance Group name: DLI-EDT-DemoConsumer
*	Spark deployment directory: /cwslocal/demoexec/dli-edt-democonsumer
*	Execution user: demoexec

<img src="images/image071.png" width="50%">

6.	Click on the Spark configuration link as shown in the picture above as well.

7.	In the “search” field, search for Java, and then fill in the JAVA_HOME environment variable with a proper directory that holds a java system of yours, for example: /usr/lib/jvm/jre-1.8.0

<img src="images/image072.png" width="50%">

8.	Then look for “SPARK_EGO_APP_SCHEDULE_POLICY” and change it to “fairshare”. 

<img src="images/image073.png" width="50%">

9.	Click on “Save” as shown above.

10.	Scroll down to the “Consumer” section and click on the standard consumer name that the process would try to create:

<img src="images/image074.png" width="50%">

11.	Click on the “X” for “DLI-EDT-DemoConsumer”:

<img src="images/image075.png" width="50%">

12.	Now select the “DemoConsumer” consumer and create a child consumer named “DLI-EDT-DemoConsumer”:

<img src="images/image076.png" width="50%">

13.	Click on “Create” and then on “Select”.

14.	Your new consumer should look like what’s show below:

<img src="images/image077.png" width="50%">

15.	Scroll down to the “Resource Groups and Plans” section and change “Spark Executors (GPU slots):” to the GPUHosts resource group. Do not change anything else.

<img src="images/image078.png" width="50%">

16.	Click on “Create and Deploy Instance Group”.

17.	Watch as your instance group gets deployed.


## Step 9: Create an instance group for Deep Learning Impact 

1.	Go to “Workload -> Instance Groups”:

<img src="images/image149.png" width="50%">

2.	Click on “New”:

<img src="images/image150.png" width="50%">

3.	Click on “Templates”:

<img src="images/image69.png" width="50%">

4.	Select “Use” for the dli-sig-template template:

<img src="images/image151.png" width="50%">

5.	Fill in the following information:
*	Instance Group name: DLI-DemoConsumer
*	Spark deployment directory: /cwslocal/demoexec/dli-democonsumer
*	Execution user: demoexec

<img src="images/image079.png" width="50%">

6.	Click on the Spark configuration link as shown in the picture above as well.

7.	In the “search” field, search for Java, and then fill in the JAVA_HOME environment variable with a proper directory that holds a java system of yours, for example: /usr/lib/jvm/jre-1.8.0

<img src="images/image072.png" width="50%">

8.	Click on “Save” as shown above.

9.	Scroll down to the “Consumer” section and click on the standard consumer name that the process would try to create:

<img src="images/image080.png" width="50%">

10.	Click on the “X” for “DLI-DemoConsumer”:

<img src="images/image081.png" width="50%">

11.	Now select the “DemoConsumer” consumer and create a child consumer named “DLI-EDT-DemoConsumer”:

<img src="images/image082.png" width="50%">

12.	Click on “Create” and then on “Select”.

13.	Your new consumer should look like what’s show below:

<img src="images/image083.png" width="50%">

14.	Scroll down to the “Resource Groups and Plans” section and change “Spark Executors (GPU slots):” to the GPUHosts resource group. Do not change anything else.

<img src="images/image078.png" width="50%">

15.	Click on “Create and Deploy Instance Group”.

16.	Watch as your instance group gets deployed.


## Exercises / Tests

### Exercise 1: Submitting a spark-based workload

This first session instructs you on how to access the conductor cluster, log onto it, locate the spark instance group and submit an application.

#### Downloading the test code

1.	Go to https://ibm.box.com/s/wnkyp42a7yxnq3zm3aji4icad7q014qy and download the file spark_apps.tar.gz

2.	Send file spark_apps.tar.gz to /cwsshare/demoexec/ on either node of your cluster. Use your favorite SCP client to do so. Remember to use “demoexec” as the user when transferring the file.

3.	As demoexec, uncompress the file:
*	cd /cwsshare/demoexec/            (read NOTE below)
*	chown demoexec:demoexec spark_apps.tar.gz   (in case you copied as some other user)
*	tar xvzf spark_apps.tar.gz

NOTE: if you’re running this in a lab environment with more students, create a folder for yourself within /cwsshare/demoexec/ and work from within it.

#### Connecting to the Conductor cluster

Conductor’s interface is reached by connecting your browser (chrome / firefox / IE) to its Web User Interface. Follow the next few steps in order to accomplish it.

4.	Open your browser and use the URL provided to you to access your Conductor cluster.  You should see a logon screen similar to what’s displayed below.

<img src="images/image084.png" width="50%">

5.	If your Conductor cluster uses SSL certificates and you’re getting SSL certificate errors, please call out to the instructor to get that fixed before you continue. A misconfigured SSL environment between your laptop and the cluster might cause you trouble down the road. The SSL certificate is under /opt/ibm/spectrumcomputing/security named cacert.pem. In order to install that certificate into your browser, download it do your workstation (use scp, filezilla, which ever scp client you have) and follow the instructions from step 7 of this link: 
*	https://www.ibm.com/support/knowledgecenter/en/SSZU2E_2.3.0/manage_cluster/security_https_pmc_enabling_dev.html


6.	Log onto the web interface by using the credentials provided to you.

7.	Once logged on, navigate to the list of instance groups by following the path: Workloads -> Spark -> Spark Instance Groups

<img src="images/image085.png" width="50%">

8.	You should see a list of instance groups you might use. The exact ones vary from environment to environment, depending on the use cases set up by IBM for you. You should see a list similar to what’s shown below.

<img src="images/image086.png" width="50%">

This first exercise will focus only on the instance groups created for the sake of submitting spark batch jobs. The Notebook and DLI instance groups will be used at a later point in time.

9.	Select the spark instance group by clicking on its name. Once you click on the instance group, you should see something similar to:

<img src="images/image087.png" width="50%">

10.	Click on the applications tab of your instance group. You should see something similar to what’s shown below.

<img src="images/image088.png" width="50%">

11.	Click on the “Run application” button, and you should be given the prompt below:

<img src="images/image089.png" width="50%">

You’ll be submitting a simple python spark code that estimates the value of pi. 

For illustrating this guide, the author placed the pi.py program under /cwsshare/demoexec/spark_apps/pi.py. So, he used this full path in the “Other options” screen, as shown below.

<img src="images/image090.png" width="50%">

12.	Click on the submit button

This brings you back to the previous screen with the list of running, submitted, finished or failed applications. Locate your application and check its status. Use the refresh button at the top every one or two seconds to check the status of your application in real time.

<img src="images/image091.png" width="50%">

13.	Once your job finishes, click on it to open an overview of it. You should see something similar to the figure below.

<img src="images/image092.png" width="50%">

Any spark program that sends anything to the standard output will have the last few lines of it displayed as shown above. Our pi.py program last statement is printing the result for estimating pi. As you can see above, the result was 3.149240. Remember, this program is just a spark example that uses spark to estimate the value of pi. The correct value should be closer to 3.1415926535897932….

If you click on the download arrow to the right of the standard output, you can download the whole standard output to your laptop.

14. Click on the “Resource Usage” tab if you’re interested in checking how many resources your program used.

<img src="images/image093.png" width="50%">

You can also submit a program by navigating the Web Interface through “Workload -> Spark -> My applications and Notebooks”, as shown below:

<img src="images/image094.png" width="50%">

15.	Now try to schedule an application to run at any given time. Go back to step 10, but this time click on the “Application Schedules” tab, as shown below.

<img src="images/image095.png" width="50%">

16.	Use the “Schedule application” button to schedule the pi.py application to run. Give it a name and select a time for it to run in a few minutes from now, then click next.

<img src="images/image096.png" width="50%">

17.	Fill out the path to the pi.py application similarly to step 11.

<img src="images/image097.png" width="50%">

18.	After your application runs, it will be accessible back in the “Submitted Applications” tab. Click on that job and validate on the right-column info that it was the one you scheduled under the name you used for it (in the  example here, Scheduled-Pi).

END of the pi.py exercise. But keep going, we’re not done just yet :-D

19.	Let’s now submit a wordcount.py application that counts how many times a given word appears on a file. The application you’ll submit is called wordcount.py and uses the text.txt file as an argument. Go back to step 10 to open the application submission window. This time, fill it out with two arguments: the full path of where your wordcount.py file is on shared storage and the full path of where your text.txt file is on shared storage:

<img src="images/image098.png" width="50%">

20.	Submit the application and keep clicking on the “refresh” button. Notice how many CPU slots it uses, but also notices that it doesn’t use any GPU slots.

<img src="images/image099.png" width="50%">

21.	Once finished, click on the application link and check the output. Feel free to download the full standard output if you wish.

<img src="images/image100.png" width="50%">

Finally, for this first part of the exercise guide, you’ll run the GPU version of the wordcount example. 

The application name is now wordcount_gpu.py, and it requires two arguments: the first is the text file and the second is an *UNEXISTING* output directory where the result will be sent to. This program does not output to standard output.

22. Go back to step 10 and this time input the full path to the wordcount_gpu.py file, the full path to the text.txt file, and the full path to the output directory. Remember, they all need to be on the shared filesystem.

<img src="images/image101.png" width="50%">

23.	After submitting your application, keep clicking on the “refresh” button and check how many slots it uses. Notice that this time this application will consume GPU slots.

<img src="images/image102.png" width="50%">

24. Compare the python code between wordcount.py and wordcount_gpu.py to see how wordcount_gpu.py is requesting spark for a GPU resource. Remember, it’s up to your code to request for a GPU slot in the cluster.

Congratulations! You’re done with the first exercise! Feel free to play around and try to submit other spark-based applications you may have handy.

### Exercise 2: Using Jupyter Notebooks in Conductor

Conductor is able to manage Jupyter Notebooks within it and makes it easy to access them. For the sake of this exercise, we’ll open Jupyter, upload a notebook to it and run it.

1.	First, access the Conductor web interface and follow the panes “Workload” -> Spark -> My Applications and Notebooks”, as seen below.

<img src="images/image103.png" width="50%">

2.	You will see a list of your submitted applications, but also notice that there is a green button on the top right which reads “Open Notebook”, as shown in the figure below.

<img src="images/image104.png" width="50%">

3.	Click on the “Open Notebook” pane. In case you have more than one Jupyter instance available to you, Conductor shows you the list of the available ones. For this exercise, in case you have more than one, select the Jupyter 5.4.0 one from instance group Notebook-DemoConsumer.

<img src="images/image105.png" width="50%">

4.	Once you open your notebook, notice that a new browser window opens. Its URL points to the compute node where your Jupyter instance is running along with a port number.

<img src="images/image106.png" width="50%">

5.	Log on with the username and password you’ve been using for Conductor.

6.	Once you open it, you should see an interface such as the one below.

<img src="images/image107.png" width="50%">

7.	Download an example notebook from https://ibm.box.com/shared/static/v5jk857igxd7s5u71oya57pvlfthrew8.ipynb and save it to your workstation.

8.	Upload that example notebook to files list. Click the “Upload button” and upload it to Jupyter.

<img src="images/image108.png" width="50%">

9.	Don’t forget to click the upload button, or your notebook won’t be sent to Jupyter.

<img src="images/image109.png" width="50%">

10.	Click on your notebook to open it, then run through it. 

<img src="images/image110.png" width="50%">

Notice: if you’re behind a proxy or your system has no access to the internet, call out to the instructor for instructions.

The notebook itself uses the MNIST dataset (images of numbers) to create a neural network, train it with that dataset, test its trained accuracy, and then infer on some additional images. Understanding the algorithm on its own is not part of the scope of this exercise. The goal is simply to have you open and use a notebook.

11.	Once done running the notebook, go back to Jupyter’s main interface by clicking on the Jupyter logo:

<img src="images/image111.png" width="50%">

12.	Turn off your notebook kernel by selecting the notebook and clicking on “Shutdown”:

<img src="images/image112.png" width="50%">

13.	Now, create a new notebook instance by following “New -> Python3” on the drop-down box on the right:

<img src="images/image113.png" width="50%">

14.	Feel free to test anything you wish in there, such as importing diverse WML-CE frameworks such as tensorflow, torch, caffe, and as well as RAPIDS frameworks such as cudf and cuml.

<img src="images/image114.png" width="50%">

Congratulations, you have completed exercise 2! Feel free to upload notebooks of your own into your Jupyter environment and play around with them.

### Exercise 3: Using datasets and models in DLI

This third session instructs you on how to navigate the DLI interface to import datasets, models and train those. It also guides you through hyperparameter search.

1.	Download the models and datasets from https://ibm.box.com/shared/static/ltjccubutz5526d9yn3py3ejhoejurws.gz to your workstation.

The official web-site for the models is https://git.ng.bluemix.net/ibmconductor-deep-learning-impact . This link contains many more models for the supported frameworks.

2.	Copy the tarball to /cwsshare/demoexec on either node. Use demoexec as the user. Then, uncompress the tarball. Read the NOTE below if you’re using this environment as a multi-student Lab.

*	chown demoexec:demoexec models_datasets.tar.gz
*	tar xvzf models_datasets.tar.gz

This creates a folder named /cwsshare/demoexec/dli_datasets_models.

NOTE: if you’re working with an environment with lots of other students, create a folder for yourself only, such as /cwsshare/demoexec/student then place and use the tarball inside it.

3.	Log onto Conductor and use the Workload tab to navigate to Deep Learning as shown below:

<img src="images/image115.png" width="50%">

4.	You’ll get to the DLI interface where you can work with datasets and models:

<img src="images/image116.png" width="50%">

Use the location of the datasets and models provided to you by your instructor. There’s a dataset and a model for Cifar10 for both Tensorflow and Caffe. Familiarize yourself with the location of these files in the filesystem. In this example, step 2 instructed you to place those under /cwsshare/demoexec/dli_datasets_models.

5.	User egoadmin has to be part of the execution user group and have proper permission to read the models and datasets. On all nodes, as root, run:

*	usermod -a -G demoexec,<other groups egoadmin already belongs to> egoadmin

6.	Once in the datasets tab, click on “New” and select the proper dataset format. For this first piece of the exercise, we’ll be using Tensorflow Records:

<img src="images/image117.png" width="50%">

7.	Fill out the dataset information with meaningful names for the dataset name. Then make sure you select the DLI-DemoConsumer instance group to work with, and finally fill out the information for the training and test folders accordingly:

<img src="images/image118.png" width="50%">

8.	Once the dataset is created, you should see something like this:

<img src="images/image119.png" width="50%">

9.	Now, go to the “Models” tab and import the corresponding Cifar10 model for Tensorflow. Click on “New” and then “Add location”:

<img src="images/image120.png" width="50%">

10.	Input the directory folder where the Cifar10 Tensorflow model is, and make sure you select “Tensorflow” as the framework. Use a meaningful name for the model location.

<img src="images/image121.png" width="50%">

11.	Click on add as shown above, then on Next:

<img src="images/image122.png" width="50%">

12. At this step, give your model a name (use something meaningful), select the training engine you’d like to use (hint: start with single node for testing purposes), and select the dataset you created in the previous steps. Select your hyperparameters value, use a number of iterations that makes sense to a test (more people might be using the cluster  ) and use a batch size of 10. Once complete, click “Add”.

<img src="images/image123.png" width="50%">

<img src="images/image124.png" width="50%">

13.	Select your model and click on “Train”:

<img src="images/image125.png" width="50%">

14.	Select the number of workers and GPUs per worker, then click on “Start training”:

<img src="images/image126.png" width="50%">

15.	To check how your training is going, click on your model, then go to the “training” tab, and check the “Insights”:

<img src="images/image127.png" width="50%">

16.	In case you selected a set of hyperparameters that is not optimal, Conductor is able to suggest some simple optimizations even before the training ends. These are shown if you click on the red “Optimize” button as show in the previous figure. Clicking on it reveals the optimization suggestion:

<img src="images/image128.png" width="50%">

17.	Once your model finishes, check the accuracy, loss and other charts.

<img src="images/image129.png" width="50%">

18.	As a next step, try running a hyperparameter search on your model. Click on the hyperparameter tuning tab, and then “New”:

<img src="images/image130.png" width="50%">

19.	Select the algorithm type you’d like to use and choose a value for the other properties. Then, scroll down and select which hyperparameters will be part of this search (learning rate, optimizer). Then click on “start tuning”.

<img src="images/image131.png" width="50%">

<img src="images/image132.png" width="50%">

20.	Once your hyperparameter search finishes, click on “More” and then navigate to the “best” tab:

<img src="images/image133.png" width="50%">

21.	You may then decide to update the current model with the best values found or copy those onto a new model.

<img src="images/image134.png" width="50%">

22. Now try to repeat the exercise using the Caffe LMDB dataset and the Caffe model for Cifar. Go to the datasets tab as explained on step 6, click “New” and this time select LMDB as the dataset source. Then fill out the information required. Use the DLI-DemoConsumer SIG and point to your dataset’s train and test folder, similarly as to what’s shown in the figure below.

<img src="images/image135.png" width="50%">

23. Once the dataset gets created, go to the Models tab and create a new model by clicking on “New”. Click “Add location” and fill out the required info, including the path to your model. Make sure the framework selected says “Caffe”.

<img src="images/image136.png" width="50%">

24. Select your model source and click on “Next” to proceed:

<img src="images/image137.png" width="50%">

25. On the following screen, give your model a name, select the training engine, select the dataset you created for cifar using the LMDB data, select your hyperparameters, and click “Add”. For batch size, use 10. For number of iterations, mind that the more you use the longer it takes for the training to complete, but the better the accuracy might be.

<img src="images/image138.png" width="50%">

26. Next, select your model and click on “Train”

<img src="images/image139.png" width="50%">

27. Depending on whether your model uses a single node engine or a distributed training engine, select the proper parameters and then start your training.

<img src="images/image140.png" width="50%">

28. To check the status of your training, click on your model, then on the “Training” tab, then on “Insights”:

<img src="images/image141.png" width="50%">

29. Now we’re going to run a Pytorch model with the Elastic Distributed Search SIG DLI-EDT-DemoConsumer. Go to the datasets tab as explained on step 6, click “New” and this time select “Any” as the data type

<img src="images/image142.png" width="50%">

25. Next, fill in the information as follows:
	
*	Dataset name: a meaningful name of your choice
*	Create in Spark Instance Group: DLI-EDT-DemoConsumer
*	Type: COPY
*	Training folder: /cwsshare/demoexec/dli_datasets_models/datasets/cifar-Pytorch-Any/train_db
* Test folder: /cwsshare/demoexec/dli_datasets_models/datasets/cifar-Pytorch-Any/test_db

Then click the “Create” button:

<img src="images/image143.png" width="50%">

26. Now import the Cifar1o Pytorch model into DLI. Go to the “Models” tab and click “New”. Then add the location of the pytorch models as shown below using:

*	Framework: Pytorch
*	Path: the path to the model (/cwsshare/demoexec/dli_datasets_models/models/Pytorch/cifar10)

Then click on “Add”:

<img src="images/image144.png" width="50%">


27. Select the Pytorch model location and click on “Next”:

<img src="images/image145.png" width="50%">

28. Fill in the info:

*	Model name:  Cifar10-Pytorch-Model-DemoUser
*	Training engine: Elastic Distributed Training
*	Training Dataset: Cifar10-Pytorch-DemoUser
*	Choose whichever hyper parameters you want


Then click on “Add”:

<img src="images/image146.png" width="50%">

29. Select the Pytorch model and click on “Train”:

<img src="images/image147.png" width="50%">


30. As  the  max number of workers, use the total number of GPUs in your cluster, then click on “Start Training”:

<img src="images/image148.png" width="50%">


31.  To see the EDT functionality taking place, simply submit another pytorch model training and see that the first job will gracefully cede some GPUs for the second one.


