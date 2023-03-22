#!/bin/sh
export LD_LIBRARY_PATH=/home/seu/miniconda3/envs/NTIRE23/lib/python3.9/site-packages/torch/lib/../../nvidia/cublas/lib/:$LD_LIBRARY_PATH
eval "$(conda shell.bash hook)"
conda activate NTIRE23
ROOT=$(dirname $0)
mkdir -p ./results

CUDA_VISIBLE_DEVICES=0 python demo/sr_demo.py --submission-id SEU_CNIIx2 \
  --checkpoint demo/PRFDN_x2.pth \
  --scale 2 \
  --lr-dir /home/data/dataset/NTIRE23-RTSR/LR2 \
  --save-sr
CUDA_VISIBLE_DEVICES=0 python demo/sr_demo.py --submission-id SEU_CNIIx3 \
  --checkpoint demo/PRFDN_x3.pth \
  --scale 3 \
  --lr-dir /home/data/dataset/NTIRE23-RTSR/LR3 \
  --save-sr
