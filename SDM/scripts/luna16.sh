#!/bin/bash

export CUDA_VISIBLE_DEVICES=0 # CHANGE ME

DATA_DIR=../../luna-16-seg-diff-data

# Training
# export OPENAI_LOGDIR='OUTPUT/LUNA'
#mpiexec -n 8 
# python3 image_train.py --data_dir "$DATA_DIR" --dataset_mode LUNA16 --lr 1e-4 --batch_size 4 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 512 --learn_sigma True \
# 	     --noise_schedule cosine --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --use_checkpoint True --num_classes 3 \
# 	     --class_cond True --no_instance True

# Classifier-free Finetune
# export OPENAI_LOGDIR='OUTPUT/ADE20K-SDM-256CH-FINETUNE'
# mpiexec -n 8 python image_train.py --data_dir ./data/ade20k --dataset_mode ade20k --lr 2e-5 --batch_size 4 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True \
# 	     --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 151 \
# 	     --class_cond True --no_instance True --drop_rate 0.2 --resume_checkpoint OUTPUT/ADE20K-SDM-256CH/model.pt

# Testing
export OPENAI_LOGDIR='OUTPUT/LUNA16-TEST'
python3 image_sample.py --data_dir "$DATA_DIR" --dataset_mode LUNA16 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True \
       --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --num_classes 6 \
       --class_cond True --no_instance True --batch_size 2 --num_samples 1000 --model_path /vol/bitbucket/bh1511/ema_0.9999_080000.pt --results_path RESULTS/ADE20K-SDM-256CH --s 1.5
