#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -u debias.py \
	--model_name_or_path "bert-base-uncased" \
	--task_type "masked_lm" \
	--prompt_model "prefix_tuning" \
	--pre_seq_len 16 \
	--train_file "data/wikipedia-10.txt" \
	--max_seq_length 496 \
	--line_by_line \
	--bias_type "gender" \
	--cda_mode "partial" \
	--output_dir "checkpoints/bert-prefix-tune-16" \
	--do_train \
	--per_device_train_batch_size 16 \
	--learning_rate 5e-3 \
	--num_train_epochs 2 \
	--save_strategy "no" \
	--evaluation_strategy "epoch" \
	--seed 42 \
	--down_sample 0.2 \
	> run_bert_prefix_16.out 2>&1