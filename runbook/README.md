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


