#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -u task.py \
	--model_name_or_path "bert-base-uncased" \
	--task_type "coref" \
	--prompt_model "none" \
	--dataset_config_name "type1" \
	--max_seq_length 512 \
	--validation_split_percentage 5 \
	--output_dir "checkpoints/wino1-bert-fine-tune" \
	--label_names 'answers' \
	--do_train \
	--do_eval \
	--per_device_train_batch_size 16 \
	--learning_rate 5e-6 \
	--num_train_epochs 30 \
	--save_strategy "no" \
	--seed 42 \
	> wino1_bert.out 2>&1