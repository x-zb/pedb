#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -u task.py \
	--model_name_or_path "gpt2" \
	--task_type "coref" \
	--prompt_model "none" \
	--dataset_config_name "type2" \
	--max_seq_length 1024 \
	--validation_split_percentage 5 \
	--output_dir "checkpoints/wino2-gpt2-fine-tune" \
	--label_names 'answers' \
	--do_train \
	--do_eval \
	--per_device_train_batch_size 16 \
	--learning_rate 5e-5 \
	--num_train_epochs 20 \
	--save_strategy "no" \
	--seed 42 \
	> wino2_gpt2.out 2>&1 