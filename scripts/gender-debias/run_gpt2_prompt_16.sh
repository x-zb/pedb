#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python -u debias.py \
	--model_name_or_path "gpt2" \
	--task_type "causal_lm" \
	--prompt_model "prompt_tuning" \
	--pre_seq_len 16 \
	--train_file "data/wikipedia-10.txt" \
	--max_seq_length 1008 \
	--line_by_line \
	--bias_type "gender" \
	--cda_mode "partial" \
	--output_dir "checkpoints/gpt2-prompt-tune-16" \
	--do_train \
	--per_device_train_batch_size 4 \
	--gradient_accumulation_steps 2 \
	--learning_rate 5e-2 \
	--num_train_epochs 2 \
	--save_strategy "no" \
	--evaluation_strategy "epoch" \
	--seed 42 \
	--down_sample 0.2 \
	> run_gpt2_prompt_16.out 2>&1 