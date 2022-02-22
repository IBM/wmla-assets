# Configure a system for deep learning in IBM Watson Machine Learning Accelerator 1.2.3

Several of the deep learning frameworks require Anaconda, which is a data science distribution that can be used on any platform. IBM Watson Machine Learning Accelerator (WML Accelerator) 1.2.3 requires you to set up the required Anaconda environment and deep learning frameworks. The available conda script installs the correct version of Anaconda and any dependent packages and deep learning frameworks.

For more information, see: http://www.ibm.com/support/knowledgecenter/SSFHA8_1.2.3/wmla_setup_dli.html

# History
* 03/19/21: Support for Python was updated from 3.7.9 to Python 3.7.10
* 07/21/21: Rename existing conda_wmla.zip to conda_wmla_py37.zip. Add new installer conda_wmla_py38.zip for Python 3.8.10
* 02/10/22: Pin dlinsights env's setuptools to 52.0.0 to resolve conflict
* 02/22/22: Pin pymongo to 3.12.0
