import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

import base64
import json
import time
import urllib

###logon###
hostname='ma1gpu03.ma.platformlab.ibm.com'
logonUrl='https://{}:8643/platform/rest/conductor/v1/auth/logon'.format(hostname)
user='wml-user'
passwd='wml-user'
base64string = base64.encodestring('%s:%s' % (user, passwd)).replace('\n', '')
auth='Basic ' + base64string
headers = {'Authorization': auth}

print(headers)
req = requests.Session()

r = req.get(logonUrl, headers=headers, verify=False)
if not r.ok:
   print('logon failed: ', r.status_code, r.content)
else:
   print('logon to rest server succeed')

commonHeaders={'accept': 'application/json'}

###start a new tuning task###

import tarfile
import tempfile
import os
def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

startTuneUrl='https://ma1gpu03.ma.platformlab.ibm.com:9243/platform/rest/deeplearning/v1/hypersearch'
data =  {
        'modelSpec': 
        {
            'sigName': 'wml-ig',
            'args': '--exec-start XGboost --cs-datastore-meta type=fs \
                     --gpuPerWorker 1 --model-main main.py \
                     --model-dir xgb-model'
        },
'algoDef':
        {
            'algorithm': 'Random',
            'maxRunTime': -1,
            'maxJobNum': 10,
            'maxParalleJobNum': 2,
            'objectiveMetric' : 'auc_dev',
            'objective' : 'maximize'
        },
        'hyperParams':
        [
             {
                 'name': 'learning_rate',
                 'type': 'Range',
                 'dataType': 'DOUBLE',
                 'minDbVal': -3,
                 'maxDbVal': 0,
                 'power': 10,
             },
             {
                 'name': 'num_rounds',
                 'type': 'Range',
                 'dataType': 'INT',
                 'minIntVal': 1,
                 'maxIntVal': 1000,
                 'step': 1,
             },
             {
                 'name': 'max_depth',
                 'type': 'Range',
                 'dataType': 'INT',
                 'minIntVal': 1,
                 'maxIntVal': 14,
                 'step': 1,
             },
             {
                 'name': 'lambda',
                 'type': 'Range',
                 'dataType': 'DOUBLE',
                 'minDbVal': -2,
                 'maxDbVal': 5,
                 'power': 10,
             },
             {
                 'name': 'colsample_bytree',
                 'type': 'Range',
                 'dataType': 'DOUBLE',
                 'minDbVal': 0.01,
                 'maxDbVal': 1.0,
                 'step': 0.01,
             },


         ]
    }

mydata={'data':json.dumps(data)}

MODEL_DIR_SUFFIX = ".modelDir.tar"
tempFile = tempfile.mktemp(MODEL_DIR_SUFFIX)
make_tarfile(tempFile, '/home/tpa/hpo/xgb-model')
files = {'file': open(tempFile, 'rb')}
create = req.post(startTuneUrl, headers=commonHeaders, data=mydata, files=files, verify=False)
if not create.ok:
   print('submit tune job failed: code=%s, %s'%(create.status_code, create.content))
else:
   print('submit tune job succeed with hponame: %s'%create.json())


######

import time

hpoName = create.json()
getHpoUrl = 'https://{}:9243/platform/rest/deeplearning/v1/hypersearch/{}'.format(hostname, hpoName)
res = req.get(getHpoUrl, headers=commonHeaders, verify=False)
if not res.ok:
    print('get hpo task failed: code=%s, %s'%(res.status_code, res.content))
else:
    json_out=res.json()

    while json_out['state'] in ['SUBMITTED','RUNNING']:
        print('Hpo task %s state %s progress %s%%'%(hpoName, json_out['state'], json_out['progress']))
        time.sleep(10)
        res = req.get(getHpoUrl, headers=commonHeaders, verify=False)
        json_out=res.json()

    print('Hpo task %s completes with state %s'%(hpoName, json_out['state']))
    print(json.dumps(json_out, indent=4, sort_keys=True))
