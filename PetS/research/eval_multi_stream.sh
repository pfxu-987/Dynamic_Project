#!/bin/bash
model=bert_base

#CUDA_VISIBLE_DEVICES=0 nsys profile --stats=true python python_scripts/cuda_test_2.py --exp_name multi_stream --model $model --log_dir exp_results/pets/$model
CUDA_VISIBLE_DEVICES=0 python python_scripts/cuda_test_1.py --exp_name multi_stream --model $model --log_dir exp_results/pets/$model