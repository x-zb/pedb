#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -u debias_adapter.py \
	--model_name_or_path "bert-base-uncased" \
	--task_type "masked_lm" \
	--train_file "data/wikipedia-10.txt" \
	--max_seq_length 512 \
	--line_by_line \
	--bias_type "race" \
	--cda_mode "partial" \
	--output_dir "checkpoints/race-bert-adapter-rf4" \
	--do_train \
	--per_device_train_batch_size 16 \
	--learning_rate 5e-4 \
	--num_train_epochs 2 \
	--save_strategy "no" \
	--evaluation_strategy "epoch" \
	--seed 42 \
	--down_sample 0.2 \
	--adapter_config "pfeiffer" \
	--adapter_reduction_factor 4 \
	> race_run_bert_adapter_rf4.out 2>&1 