# IBM Watson Machine Learning 2.0 Add-on Installation

Assumption : In order to deploy IBM Watson Machine Learning 2.0 it requires IBM Watson Studio Local 2.0 base installation to be already deployed on the cluster. 
So the below steps outlined are followed based on this assumption.

##  Installation/Configuration of Watson Machine Learning 2.0 

1. Download the IBM Watson Machine Learning add-on Evaluation software (wml_module.tar) from the <IBM Software Repository>. This downloadable package is around 16GB in size and requires IBM ID to download.

2. You can download the Watson Machine Learning software add-on package to the /ibm/InstallPackage/modules directory. On successful download the file exists as /ibm/InstallPackage/modules/wml_module.tar  

3. Now we are all set deploy the IBM Watson Machine Learning add-on package. Go to /ibm/InstallPackage/components directory and execute the below command

        cd /ibm/InstallPackage/components
        ./deploy.sh /ibm/InstallPackage/modules/ wml_module.tar
   It asks for License acceptance and once you accept the license terms the above deployment might take around 30 minutes to complete.

4. Post successful IBM Watson Machine Learning add-on package, execute following command to verify if the add-on package deployment was successful 
        
        helm list --tls | grep wml 
   The above command will list the WML package with deployment Status (if its Successful it would display DEPLOYED).
   
<img src="OneAI_install_images/image001.png" width="50%">


#  IBM Watson Machine Learning Accelerator 1.2.1 Installation

## Steps

#### Step 1. Download IBM Watson Machine Learning Accelerator 1.2.1 Evaluation


* Download the IBM Watson Machine Learning Accelerator Evaluation software from the IBM Software Repository. This is a 4.9 GB download and requires an IBM ID.

#### Step 2. Read and follow "Set up your system" and make sure your system meets all the perquisites before you proceed to install Watson Machine Learning Accelerator 1.2.1. 

* DLI_SHARED_FS is supported in NFS3. You would need to export DLI_SHARED_FS when installing Watson Machine Learning Accelerator 1.2.0.
* DLI_RESULT_FS and DLI_DATA_FS are supported in NFS4. You would need to export DLI_RESULT_FS and DLI_DATA_FS when installing Watson Machine Learning Accelerator 1.2.1.

#### Step 3. Install Watson Machine Learning Accelerator 1.2.1 using the instructions given in the IBM Knowledge Center. 

#### Step 4. Download and install manadatory iFix for Watson Machine Learning Accelerator -> https://www-945.ibm.com/support/fixcentral/swg/selectFixes?parent=ibm%7EOther%20software&product=ibm/Other+software/IBM+PowerAI+Enterprise&release=1.2.0&platform=All&function=all


#### Step 5. Configuration of Watson Machine Learning Accelerator with Watson Studio token authentication 

Get DLI_JWT_SECRET_KEY from Watson Studio Local and save DLI_JWT_SECRET_KEY to a shared file system accessible across all nodes. You would need to export DLI_JWT_SECRET_KEY file path when installing Watson Machine Learning Accelerator 1.2.1.

1. `$> wget --no-check-certificate https://<watson_studion_local_host>:<ws_port>/auth/jwtcert`
2. `$> openssl x509 -pubkey -in jwtcert -noout >ws_local.pem`
3. `$> source /opt/ibm/spectrumcomputing/profile.platform`
3.	Edit dlpd.conf file ($EGO_TOP/dli/conf/dlpd/dlpd.conf) and update DLI_JWT_SECRET_KEY value to the location of newly generated ws_local.pem file

     *	Copy and paste DLI_JWT_SECRET_KEY to /shared_nfs4/icp4d_key.pem
     *  `# chmod 777 /shared_nfs4/icp4d_key.pem`
4. restart dlpd service
```
$> source /opt/wmla/ego_top/profile.platform
$> egosh user logon -u Admin -x Admin
$> egosh service stop dlpd
$> sleep 5
$> egosh service start dlpd
```

#### Step 6. Create local OS user "wml-user” across all nodes. This will be the instance groups’ execution user. Create local OS user where the name corresponds to the Watson Studio Local User who will submit workload.  Please make 

`useradd -m -u 20005 wml-user`

#### Step 7. Ensure that GPU is enabled for deep learning workloads. If you do not have GPU enabled, enable GPU by running one of these scripts on the master host command line interface:

* To run with user interaction: 
    ```
    # $EGO_TOP/conductorspark/2.3.0/etc/gpuconfig.sh enable
    ```
* To run without user interaction:
    ```
    # $EGO_TOP/conductorspark/2.3.0/etc/gpuconfig.sh enable --quiet -u <username> -x <password>
    ```
* Navigate to the Watson Machine Learning Accelerator GUI and logon.

<img src="OneAI_install_images/image002.jpg" width="50%">

#### Step 8. Create GPU resource group.

* Click on “Resources”> “Resource Planning (Slot)” > “Resource Groups”

<img src="OneAI_install_images/image003.jpg" width="50%">

* Click on “Global Actions”> “Create a Resource Group”

<img src="OneAI_install_images/image004.jpg" width="50%">

* Fill in the “Resource group name”, “Advanced Formula” to “ngpus” and then click “Create”

<img src="OneAI_install_images/image005.jpg" width="50%">

#### Step 9. Create wml-ig.

* Click on “Workload”> “Spark” > “Spark Instance Groups”

<img src="OneAI_install_images/image006.jpg" width="50%">

* Click on “Create a Spark instance Group”

<img src="OneAI_install_images/image007.jpg" width="50%">

* Click on “Templates”

<img src="OneAI_install_images/image008.jpg" width="50%">

* Click on “wmla-ig-template-2-3-1 use”

<img src="OneAI_install_images/image009.jpg" width="50%">

* Fill in “Instance group name” to “wml-ig”, “Spark deployment directory”, “Execution user for instance group” to “wml-user”. Ensure “Spark deployment directory” is accessible to “wml-user” across all nodes.

<img src="OneAI_install_images/image010.jpg" width="50%">

* Select the GPU resource group you just created in step 8 of this tutorial and click “Create and Deploy Instance Group”

<img src="OneAI_install_images/image011.jpg" width="50%">

#### Step 10: Create wml-ig-edt.

* Click on “Workload”> “Spark” > “Spark Instance Groups”

<img src="OneAI_install_images/image006.jpg" width="50%">

* Click on "New"
<img src="OneAI_install_images/image012.jpg" width="50%">

* Click on “Templates”

<img src="OneAI_install_images/image008.jpg" width="50%">

* Click on “wmla-ig-edt-template-2-3-1 use”

<img src="OneAI_install_images/image013.jpg" width="50%">

* Fill in “Instance group name” to “wml-ig-edt”, “Spark deployment directory”, “Execution user for instance group” to “wml-user”. Ensure “Spark deployment directory” is accessible to “wml-user” across all nodes.   Please make sure the parent directory "/opt/wmla-ig" pre-exists and "wml-user" has write access to this parent directory.

<img src="OneAI_install_images/image014.jpg" width="50%">

* Select the GPU resource group you just created in step 8 of this tutorial and click “Create and Deploy Instance Group”

<img src="OneAI_install_images/image011.jpg" width="50%">

#### Step 11. Create EGO user corresponding to all OS user created in step 6 of this tutorial

* Click on “System & Services” > “Users” > “Accounts” 

<img src="OneAI_install_images/image015.jpg" width="50%">

* Click on “Global Actions” > “Create New User Account”

<img src="OneAI_install_images/image016.jpg" width="50%">

* Fill in the “User Name”, “Password” and click create

#### Step 12. Step Assign the created EGO user consumer role for wml-ig and wml-ig-edt

* Click on “System & Services” > “Users” > “Roles” 

* Click on “Consumer User” > “wml-user” > “wml-ig” and “wml-ig-edt” and click “Apply”

<img src="OneAI_install_images/image017.jpg" width="50%">

#### Step 13a. Set the variable METRICS_STREAMING=Y in $EGO_CONFDIR/dli/conf/dlpd/dlpd.conf

#### Step 13b. Enable EGO_ENABLE_CONSUMER_LEVEL_EXCLUSIVE.  In Master host command line interface, do the following:

* Open the $EGO_CONFDIR/ego.conf file for editing.
* `EGO_ENABLE_CONSUMER_LEVEL_EXCLUSIVE=Y`
* Save your changes
* Restart the cluster:
    ```
    egosh service stop all
    egosh ego shutdown
    egosh ego start all
    ```
#### Step 14. Enable the following so that for non-EDT jobs, if a job allocates 4 GPUs then it allocates the whole host exclusively. Configure wml-ig and wml-ig-edt GPU resource plan.
 
* Click on “Resource”> “Resource Group Planning (Slot)” > “Resource Plan”

<img src="OneAI_install_images/image018.jpg" width="50%">

* Configure resource plan for wml-ig and wml-ig-edt as following:

<img src="OneAI_install_images/image019.png" width="50%">

There is known issue -> https://www.ibm.com/support/knowledgecenter/en/SSFHA8_1.2.0/release_note_s/known_issues_limitations.html

  ```
Cluster fails to respond when saving changes to a Spark instance group that was created using the wmla-ig-template-2-3-1 template
    When configuring a Spark instance group and setting EGO_ENABLE_CONSUMER_LEVEL_EXCLUSIVE=Y in the ego.conf file, the following error message is displayed when saving your changes: Error 500: javax.servlet.ServletException: The cluster is not responding.

    To resolve this issue, try saving the resource group plan again.
  ```



#### Step 15. Set “Reclaim Grace Period” for wml-ig and wml-ig-edt.

* Click on “Resources”> “Consumers”

<img src="OneAI_install_images/image020.jpg" width="50%">

* Set “Reclaim Grace Period” to “596 hours” in consumer level for wml-ig and for each child consumer belonging to wml-ig as shown below:

<img src="OneAI_install_images/image021.jpg" width="50%">

* Set “Reclaim Grace Period” to “120 seconds” in consumer level for wml-ig-edt as shown below:

<img src="OneAI_install_images/image022.jpg" width="50%">

#### Step 16. Configure GPU mode to exclusive across all nodes. In command line interface run:

* `nvidia-smi -c 1` to set GPU mode to exclusive process mode
* `nvidia-smi` command to ensure GPU mode is set to exclusive process mode
* restart ego

<img src="OneAI_install_images/nvidia-smi.png" width="50%">

#### Step 17:   Connect Watson Machine Learning with Watson Machine Learning Accelerator

1. Access  Watson Machine Learning host (via SSH) on Master node

2. `$> cd /ibm/InstallPackage/components/modules/wml`

3. run `$> ./updateWMLClusterdetails.sh <wmla_host> <wmla_ port> <wmla_default_ig> <wmla_default_edt_ig> <wml_external host adress>`
  where:
  * <wmla _host> – is the ip address that can be accessed from WML cluster in WMLA cluster master node
  * <wmla_ port> - the port exposed by WMLA for DLI rest API. Usually this is 9243
  * <wmla_default_ig> - the instance group name created in WMLA for regular jobs. Usually this value is wml-ig
  * <wmla_default_edt_ig> - ihe instance group created in WMLA for Elastic Distributed Training jobs. Usually this value is wml-ig-edt
  * <wml_external_host> - The external host name or IP address of WML that can be accessed from WMLA.(This is the same as Watson Studio host name)


      Example:
      ```
      /updateWMLClusterdetails.sh 10.15.112.32 9243 wml-ig wml-ig-edt 169.224.121.54
      ```
      or:
      ```
      ./updateWMLClusterdetails.sh some-wmla-host.ibm.com 9243 wml-ig wml-ig-edt some-ws-host.ibm.com
      ```










