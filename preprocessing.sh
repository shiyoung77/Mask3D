#!/bin/bash

python -m datasets.preprocessing.scannet_preprocessing preprocess \
    --data_dir="/mnt/evo/dataset/ScanNet" \
    --save_dir="/mnt/evo/dataset/processed_scannet" \
    --git_repo="/home/lsy/software/ScanNet/" \
    --scannet200=true
