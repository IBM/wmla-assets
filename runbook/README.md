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

1.	Go to “Workload -> Spark -> Spark Instance Groups”:

<img src="images/image055.png" width="50%">

2.	Click on “New”:

<img src="images/image068.png" width="50%">

3.	Click on “Templates”:

<img src="images/image69.png" width="50%">

4.	Select “Use” for the dli-sig-template template:

<img src="images/image070.png" width="50%">

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

(To be completed)
