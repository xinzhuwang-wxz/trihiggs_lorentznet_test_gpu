#!/bin/bash
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=12345 
source /lustre/collider/wangxinzhu/conda_xinzhu.env
conda activate pytorch_trihiggs
python ./train.py
