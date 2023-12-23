### README for Elastic Distributed Inference models

* Example request and response:
```
(echo -n '{"data_type":"image:raw_data", "data_value":[ { "key":"uri", "value":"';base64 -w 0 ./cat.jpg;echo '" } ], "action_type":"Classification", "attributes":[] }') | curl -k -X POST -d @- -H "Authorization: Bearer `dlim config -s`" https://wmla-inference-bjwjliao.apps.wml1x210.ma.platformlab.ibm.com/dlim/v1/inference/tf-cifar10
{"predictions": [{"keys": ["uri"], "results": {"dense_1": [[24.300403594970703, 210.4494171142578, -282.60980224609375, -12.286184310913086, -605.46630859375, -92.38099670410156, -10.676700592041016, -351.67230224609375, 139.08152770996094, 327.57562255859375]]}}]}
```
