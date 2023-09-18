#!/bin/bash
  
torchrun --nnodes $WMLA_NWORKER --nproc-per-node=$WMLA_WORKER_NDEVICE --node_rank=$NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT $@