To deploy this model:
1. put PyTorch models in the same directory with kernel.py
2. use `dlim model deploy -p <your_working_dir>` to deploy model

A simple notebook to send inference request:
```
import json
import numpy as np
from PIL import Image
import base64
with open("000000581781.jpg", 'rb') as f:
    base64_str = base64.b64encode(f.read())
my_data = {
    "id": 1,
    "inputs": [
     {
         "name": "gpu_0/data",
         "shape": [478, 640],
         "datatype": "BYTES",
         "data": str(base64_str, encoding = "utf-8")
    }],
    "outputs":[]
}
import requests

url="http://<wmla-inference route>/dlim/v1/inference/<model_name>"
headers = {"X-User-Token": "<you_token>"}
res=requests.post(url, headers=headers, data=json.dumps(my_data), verify=False)
print("The response: {}".format(res.content))
```

The response will be like this:
```
{'id': 1, 'outputs': [{'name': 'output0', 'datatype': 'FP32', 'shape': [1, 10], 'data': [0.07383891940116882, -1.1525914669036865, 0.8941959142684937, 0.7748421430587769, 0.5024958848953247, 0.25152453780174255, -0.10158365964889526, 0.24579831957817078, -0.12934023141860962, -0.6191686391830444]}]}
```
