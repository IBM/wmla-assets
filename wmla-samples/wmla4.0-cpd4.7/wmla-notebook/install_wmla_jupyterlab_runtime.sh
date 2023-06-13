#!/bin/bash

R='\033[0;31m'
G='\033[0;32m'
NC='\033[0m'
SUCCESS="${G}OK${NC}"
FAIL="${R}FAIL${NC}"

usage() {
  echo "Usage: $0 [-u username] [-x password] [-c cpd_host]"
  echo "  -u: The username to use for authentication."
  echo "  -x: The password to use for authentication."
  echo "  -c: The CPD console hostname to access."
  echo "      The hostname should be the output of cmd \"oc get route cpd -o jsonpath={.spec.host}\""
  echo ""
  echo "If no options are provided, the user will be prompted to input the username, password, and hostname."
}

while getopts "u:x:c:h" opt; do
  case $opt in
    u) username=$OPTARG ;;
    x) password=$OPTARG ;;
    c) cpd_host=$OPTARG ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done

if [ -z "$username" ]; then
  echo "Please enter your username:"
  read username
fi

if [ -z "$password" ]; then
  echo "Please enter your password:"
  read -s password
fi

if [ -z "$cpd_host" ]; then
  echo "Please enter the CPD hostname (the output of cmd \"oc get route cpd -o jsonpath={.spec.host}\"):"
  read cpd_host
fi

echo
if which jq >/dev/null 2>&1 ; then
  echo -e "Check jq is installed - ${SUCCESS}"
  echo
else
  echo "jq is not installed!"
  exit 1
fi

echo Validating auth using username $username to host $cpd_host
myToken=`curl -k -s https://${cpd_host}/v1/preauth/validateAuth -u ${username}:${password} | sed -n -e 's/^.*accessToken":"//p' | cut -d'"' -f1`
if [ -z "${myToken}" ]
then 
    echo -e "Validate auth - ${FAIL}"
    exit 1
fi
echo -e "Validate auth - ${SUCCESS}"

echo
echo Check existing WMLA notebook runtime configurations...
wmla_guid=`curl -s -k -H "Authorization: Bearer ${myToken}" "https://${cpd_host}/v2/runtime_definitions" | jq -r '.resources[] |select(.entity.name=="jupyter-wmla-jupyterlab-py") |.metadata.guid'`
if [ ! -z "${wmla_guid}" ]
then
    echo Found existing WMLA notebook runtime, remove it...
    output=`curl -s -k -X DELETE -H "Authorization: Bearer ${myToken}" "https://${cpd_host}/v2/runtime_definitions/${wmla_guid}"`
    if [ "$(echo ${output} |grep -c 'error')" -eq 1 ]
    then
        echo -e "Remove existing WMLA notebook runtime - ${FAIL}"
        echo Please check the error msg and remove the runtime manually to continue.
        exit 1
    else
        echo -e "Remove existing WMLA notebook runtime - ${SUCCESS}"
    fi
else
    echo Existing WMLA notebook runtime not found.
fi

echo
echo Query the default WS notebook runtime configurations...
ws_guid=`curl -s -k -H "Authorization: Bearer ${myToken}" "https://${cpd_host}/v2/runtime_definitions" | jq -r '.resources[] |select(.entity.name=="jupyter-231l-py") |.metadata.guid'`
echo WS default runtime guid: $ws_guid
if [ -z "${ws_guid}" ]
then
    echo -e "Query the default WS notebook runtime configurations - ${FAIL}"
    exit 1
fi
echo -e "Query the default WS notebook runtime configurations - ${SUCCESS}"

echo 
echo Create WMLA notebook runtime configurations...
curl -s -k -H "Authorization: Bearer ${myToken}" "https://${cpd_host}/v2/runtime_definitions/${ws_guid}?include=launch_configuration" | jq -r '.entity | .name="jupyter-wmla-jupyterlab-py" | .launch_configuration.env += [{
        "name": "_userdefined_PYTHONPATH",
        "value": "/cc-home/_global_/python-3.10:/cc-home/_global_/python-3:/opt/wmla/site-packages/ibm_wmla_lib/ziplibs/fabric.zip:/opt/wmla/site-packages/ibm_wmla_lib/ziplibs/hiddenlayer-0.2.zip:/opt/wmla/site-packages/ibm_wmla_lib/ziplibs/nest-asyncio-1.3.2.zip:/opt/wmla/site-packages/ibm_wmla_lib/ziplibs/py4j-0.10.9-src.zip:/opt/wmla/site-packages/ibm_wmla_lib/ziplibs/pyspark.zip:/opt/wmla/site-packages/ibm_wmla_lib/data:/opt/wmla/site-packages/ibm_wmla_lib/train:/opt/wmla/libs/fabric"
      },
      {
        "name": "_userdefined_LD_LIBRARY_PATH",
        "value": "/opt/wmla/libs/fabric:/opt/ibm/dsdriver/lib:/opt/oracle/lib:/opt/conda/envs/Python-3.10-Premium/lib/python3.10/site-packages/tensorflow"
      },
      {
        "name": "_userdefined_EXEC_TYPE",
        "value": "notebook"
      },
      {
        "name": "_userdefined_DATA_DIR",
        "value": "/gpfs/mydatafs"
      },
      {
        "name": "_userdefined_SPARK_EGO_FABRIC_SC",
        "value": "kub"
      }] | .launch_configuration.volumes += [      {
        "volume": "mydatafs",
        "mountPath": "/gpfs/mydatafs",
        "claimName": "wmla-mygpfs",
        "subPath": "mydatafs",
        "optional": false
      },
      {
        "volume": "wmla",
        "mountPath": "/opt/wmla",
        "claimName": "wmla-mygpfs",
        "subPath": "wmla",
        "optional": false
      },
      {
        "volume": "wmla-logging",
        "mountPath": "/wmla-logging/notebook",
        "claimName": "wmla-logging",
        "subPath": "notebook",
        "optional": false
      }]' > jupyter-wmla-jupyterlab-py-server.json
echo Save configurations to jupyter-wmla-jupyterlab-py-server.json
echo -e "Create WMLA notebook runtime configurations - ${SUCCESS}"

echo
echo Upload WMLA notebook runtime configurations...
output=`curl -k -s -X POST -H "Authorization: Bearer ${myToken}" "https://${cpd_host}/v2/runtime_definitions/" -H 'Content-Type: application/json' -d @jupyter-wmla-jupyterlab-py-server.json`
echo ${output}
if [ "$(echo ${output} | grep -c 'guid')" -eq 1 ]
then
    echo -e "Upload WMLA notebook runtime configurations - ${SUCCESS}"
else
    echo -e "Upload WMLA notebook runtime configurations - ${FAIL}"
fi

echo
echo All done.
