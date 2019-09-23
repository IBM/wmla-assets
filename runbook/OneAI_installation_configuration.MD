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

#### Step 1. Install PowerAI 1.6.1 by following Knowledge Center


#### Step 2. Download IBM Watson Machine Learning Accelerator 1.2.1 Evaluation

* Download the IBM Watson Machine Learning Accelerator Evaluation software from the IBM Software Repository. This is a 4.9 GB download and requires an IBM ID.

#### Read and follow "Set up your system" and make sure your system meets all the perquisites before you proceed to install Watson Machine Learning Accelerator 1.2.1. 

* DLI_SHARED_FS is supported in NFS3. You would need to export DLI_SHARED_FS when installing Watson Machine Learning Accelerator 1.2.0.
* DLI_RESULT_FS and DLI_DATA_FS are supported in NFS4. You would need to export DLI_RESULT_FS and DLI_DATA_FS when installing Watson Machine Learning Accelerator 1.2.1.

#### Step 4. Install Watson Machine Learning Accelerator 1.2.1 using the instructions given in the IBM Knowledge Center. 

#### Step 5. Configuration of Watson Machine Learning Accelerator with Watson Studio token authentication 

Get DLI_JWT_SECRET_KEY from Watson Studio Local and save DLI_JWT_SECRET_KEY to a shared file system accessible across all nodes. You would need to export DLI_JWT_SECRET_KEY file path when installing Watson Machine Learning Accelerator 1.2.1.

1. `$> wget --no-check-certificate https://<watson_studion_local_host>:<ws_port>/auth/jwtcert`
2. `$> openssl x509 -pubkey -in jwtcert -noout >ws_local.pem`
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

#### Step 6. Create local OS user "wml-user” across all nodes. This will be the instance groups’ execution user. Create local OS user user where the name corresponds to the Watson Studio Local User who will submit workload.

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

* Fill in “Instance group name” to “wml-ig-edt”, “Spark deployment directory”, “Execution user for instance group” to “wml-user”. Ensure “Spark deployment directory” is accessible to “wml-user” across all nodes.

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

#### Step 13. Enable EGO_ENABLE_CONSUMER_LEVEL_EXCLUSIVE. In Master host command line interface, do the following:

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

#### Step 15. Set “Reclaim Grace Period” for wml-ig and wml-ig-edt.

* Click on “Resources”> “Consumers”

<img src="OneAI_install_images/image020.jpg" width="50%">

