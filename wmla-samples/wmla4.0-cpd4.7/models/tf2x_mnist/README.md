## Tensorflow minst model

### cpu training (1 worker)
```
source /opt/anaconda3/bin/activate dlipy3-cpu
python main.py --no-cuda
```

### gpu training (1 worker, 1 gpu per worker)
```
source /opt/anaconda3/bin/activate dlipy3
export CUDA_VISIBLE_DEVICES=0
python main.py
```

### gpu training (1 worker, multiple gpu per worker)
```
source /opt/anaconda3/bin/activate dlipy3
export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py
```

### dist gpu training with 1 gpu(2 workers on same node, 1 gpu per worker)
start worker #1 on dlw10
```
source /opt/anaconda3/bin/activate dlipy3
export CUDA_VISIBLE_DEVICES=0
export TF_CONFIG='{"cluster": {"worker": ["dlw10.aus.stglabs.ibm.com:46001", "dlw10.aus.stglabs.ibm.com:46002"]}, "task": {"index": 0, "type": "worker"}}'
python main.py
```

start worker #2 on dlw10
```
source /opt/anaconda3/bin/activate dlipy3
export CUDA_VISIBLE_DEVICES=1
export TF_CONFIG='{"cluster": {"worker": ["dlw10.aus.stglabs.ibm.com:46001", "dlw10.aus.stglabs.ibm.com:46002"]}, "task": {"index": 1, "type": "worker"}}'
python main.py
```

### dist gpu training with multi gpu(2 workers on same node, 2 gpu per worker)
start worker #1 on dlw10
```
source /opt/anaconda3/bin/activate dlipy3
export CUDA_VISIBLE_DEVICES=0,1
export TF_CONFIG='{"cluster": {"worker": ["dlw10.aus.stglabs.ibm.com:46001", "dlw10.aus.stglabs.ibm.com:46002"]}, "task": {"index": 0, "type": "worker"}}'
python main.py
```

start worker #2 on dlw10
```
source /opt/anaconda3/bin/activate dlipy3
export CUDA_VISIBLE_DEVICES=2,3
export TF_CONFIG='{"cluster": {"worker": ["dlw10.aus.stglabs.ibm.com:46001", "dlw10.aus.stglabs.ibm.com:46002"]}, "task": {"index": 1, "type": "worker"}}'
python main.py
```