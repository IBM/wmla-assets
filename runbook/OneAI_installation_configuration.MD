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

### Step 1. Install PowerAI 1.6.1 by following Knowledge Center


### Step 2. Download IBM Watson Machine Learning Accelerator 1.2.1 Evaluation

* Download the IBM Watson Machine Learning Accelerator Evaluation software from the IBM Software Repository. This is a 4.9 GB download and requires an IBM ID.

