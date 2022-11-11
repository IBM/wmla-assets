#! /bin/bash

WMLA_REST_HOST=""
USERNAME=""
PASSWORD=""
DLICMD=dlicmd.py

wait_for_app_finish() {
  local appId=$1
  local finish=0

  cmd="./$DLICMD --exec-get $appId --rest-host $WMLA_REST_HOST"
  echo "INFO: $cmd"

  while [ $finish -eq 0 ]
  do
    local result=`$cmd`
    state=`echo $result | jq -r '.state'`

    #echo "DEBUG: $result"
    if [[ "$state" == *"FINISHED"* ]]; then
      echo "INFO: state=$state"
      finish=1
    elif [[ "$state" == *"ERROR"* ]]; then
      echo "ERROR: state=$state"
      finish=1
    else
      echo "INFO: state=$state"
      sleep 5
    fi
  done
}

display_usage() {
  echo -e "\nUsage: \$0 -r restHost -u username -p password\n"
  echo -e "   -r restHost          -- WMLA REST api host, e.g., wmla-console-wmla-ns.ibm.com"
  echo -e "   -u username          -- "
  echo -e "   -p password          -- "
}

# MAIN
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# if less than two arguments supplied, display usage
if [  $# -le 1 ]
then
  display_usage
  exit 1
fi

while getopts "r:u:p:" flag
do
    case "${flag}" in
        r) WMLA_REST_HOST=${OPTARG};;
        u) USERNAME=${OPTARG};;
        p) PASSWORD=${OPTARG};;
    esac
done

# MAIN
echo ""
if [ ! -d "$SCRIPT_DIR/workdir" ]; then
  mkdir $SCRIPT_DIR/workdir
fi

cd $SCRIPT_DIR/workdir
echo ""
echo "INFO: download WMLA commandline"
cmd="wget https://$WMLA_REST_HOST/ui/tools/dlicmd.py -O $DLICMD --no-check-certificate --quiet"
echo "INFO: cmd=$cmd"
wget https://$WMLA_REST_HOST/ui/tools/dlicmd.py -O $DLICMD --no-check-certificate --quiet

echo ""
echo "INFO: logon to WMLA"
cmd="./$DLICMD --logon --rest-host $WMLA_REST_HOST --username $USERNAME --password $PASSWORD"
echo "INFO: cmd=$cmd"
result=`$cmd`

echo ""
echo "INFO: submit the python script"
cmd="./$DLICMD --exec-start tensorflow --rest-host $WMLA_REST_HOST --appName my-tf-app --workerDeviceType cpu  --numWorker 1 --model-main mnist_convnet.py --model-dir ../mnist_convnet"
echo "INFO: cmd=$cmd"
result=`$cmd`
echo $result

echo ""
echo "INFO: Get app Id"
x=`echo \`expr index "$result" {\``
y=`echo ${result:$x-1}`
appId=`echo $y | jq -r '.appId'`
#appId=wmla-ns-33
echo "INFO: appId=$appId"

echo ""
echo "INFO: wait for app $appId to finish"
wait_for_app_finish $appId

echo ""
echo "INFO: get training stdout and stderr"
cmd="./$DLICMD --exec-trainlogs $appId --rest-host $WMLA_REST_HOST"
echo "INFO: cmd=$cmd"
result=`$cmd`
if [[ "$result" == *"Test loss:"* ]] && [[ "$result" == *"Test accuracy:"* ]] ; then
  echo "INFO: expect trainoutlogs contains *"Test loss:"* and *"Test accuracy:"*. OK"
else
  echo "ERROR: expect trainoutlogs contains *"Test loss:"* and *"Test accuracy:"*. Got $result. FAILED"
fi

echo ""
echo "INFO: get training result"
cmd="./$DLICMD --exec-trainresult $appId --rest-host $WMLA_REST_HOST"
echo "INFO: cmd=$cmd"
result=`$cmd`
if [[ "$result" == *"Train result is saved to dlpd-model"* ]]; then
  echo "INFO: expect trainresult contains *"Train result is saved to dlpd-model"*. OK"
else
  echo "ERROR: expect trainresult contains *"Train result is saved to dlpd-model"*. Got $result. FAILED"
fi

