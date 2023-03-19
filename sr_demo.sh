#!/bin/sh
export LD_LIBRARY_PATH=/home/seu/miniconda3/envs/NTIRE23/lib/python3.9/site-packages/torch/lib/../../nvidia/cublas/lib/:$LD_LIBRARY_PATH
eval "$(conda shell.bash hook)"
conda activate NTIRE23
ROOT=$(dirname $0)
mkdir -p ./results

CUDA_VISIBLE_DEVICES=0 python demo/sr_demo.py --submission-id SEU_CNII --checkpoint /home/data/NTIRE2022_ESR/save_models/RFDN_-10_trainLSDIRX2_Param0.27M/model/model_latest.pt.pth --scale 2 --lr-dir /home/data/dataset/DIV2K/DIV2K_valid_HR --save-sr
CUDA_VISIBLE_DEVICES=0 python demo/sr_demo.py --submission-id SEU_CNII --checkpoint /home/data/NTIRE2022_ESR/save_models/RFDN_-10_trainLSDIRX3_Param0.27M/model/model_latest.pt.pth --scale 3 --lr-dir /home/data/dataset/DIV2K/DIV2K_valid_HR --save-sr
