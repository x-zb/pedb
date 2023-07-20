#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python -u debias.py \
	--model_name_or_path "bert-base-uncased" \
	--task_type "masked_lm" \
	--prompt_model "prompt_tuning" \
	--pre_seq_len 192 \
	--train_file "data/wikipedia-10.txt" \
	--max_seq_length 320 \
	--line_by_line \
	--bias_type "race" \
	--cda_mode "partial" \
	--output_dir "checkpoints/race-bert-prompt-tune-192" \
	--do_train \
	--per_device_train_batch_size 16 \
	--learning_rate 5e-3 \
	--num_train_epochs 2 \
	--save_strategy "no" \
	--evaluation_strategy "epoch" \
	--seed 42 \
	--down_sample 0.2 \
	> race_run_bert_prompt_192.out 2>&1 &