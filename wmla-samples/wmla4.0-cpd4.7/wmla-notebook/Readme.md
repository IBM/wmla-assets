
# Install Steps
Install the Watson Machine Learning Accelerator (WMLA) elastic notebook feature into the Watson Studio (WS) notebook.

## Install WMLA notebook runtime configuration
- Download script `install_wmla_notebook_runtime.sh`
- install jq
```
yum install jq
```
- Run the script, eg:
```
bash install_wmla_notebook_runtime.sh
bash install_wmla_notebook_runtime.sh -u <user> -c <cpd_host>
bash install_wmla_notebook_runtime.sh -u <user> -x <password> -c <cpd_host>
bash install_wmla_notebook_runtime.sh -h
```

## Install WMLA Jupyterlab runtime configuration
- Download script `install_wmla_jupyterlab_runtime.sh`
- install jq
```
yum install jq
```
- Run the script, eg:
```
bash install_wmla_jupyterlab_runtime.sh
bash install_wmla_jupyterlab_runtime.sh -u <user> -c <cpd_host>
bash install_wmla_jupyterlab_runtime.sh -u <user> -x <password> -c <cpd_host>
bash install_wmla_jupyterlab_runtime.sh -h
```
