#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python -u task_adapter.py \
	--model_name_or_path "bert-base-uncased" \
	--task_type "coref" \
	--dataset_config_name "type1" \
	--max_seq_length 512 \
	--validation_split_percentage 5 \
	--output_dir "checkpoints/wino1-bert-adapter_rf48" \
	--label_names 'answers' \
	--do_train \
	--do_eval \
	--per_device_train_batch_size 16 \
	--learning_rate 5e-4 \
	--num_train_epochs 20 \
	--save_strategy "no" \
	--seed 42 \
	--adapter_config "pfeiffer" \
	--adapter_reduction_factor 48 \
	> wino1_bert_adapter_rf48.out 2>&1