# README of model pingpong

## Summary

This is a pingpong test which could be used for EDI networking performance test.

## Input

* Input format: json
* Input body:

```
{
    "data" : "12345"
}
```

## Output
* Output format: json
* Output body (if there is no error)
```
{
    "data" : "12345"
}
```

## Caller example

### WML-A 2.3.x
- 1, first to get the token for inference
```
oc project ${wmla-ns}
WMLA_CONSOLE_FQDN=$(oc get route wmla-console -o jsonpath='{.spec.host}')
curl -k -u user:password -X POST https://${WMLA_CONSOLE_FQDN}/dlim/v1/auth/token
{"access_token":"${YOUR_TOKEN}", "service_token": "${INFERENCE_TOKEN}"}
```
- 2, send a Restful inference request to EDI
```
oc project ${wmla-ns}
WMLA_INFERENCE_FQDN=$(oc get route wmla-inference -o jsonpath='{.spec.host}')
curl -k -H "X-Auth-Token: ${INFERENCE_TOKEN}" -d '{"data":"xxxxx", "seq":1}' -X POST https://${WMLA_INFERENCE_FQDN}/dlim/v1/inference/pingpong
{"data": "xxxxx", "seq": 1}
```

### WML-A 1.2.3
- 1, first to get the token for inference
```
curl -k -u user:passwd -X POST https://${YOUR_FQDN}:9000/dlim/v1/auth/token
{"access_token":"${YOUR_TOKEN}", "service_token": "${INFERENCE_TOKEN}"}
```
- 2, send a Restful inference request to EDI
```
curl -k -H "X-Auth-Token: ${INFERENCE_TOKEN}" -d '{"data":"xxxxx", "seq":1}' -X POST https://${YOUR_FQDN}:9000/dlim/v1/inference/pingpong
{"data": "xxxxx", "seq": 1}
```

