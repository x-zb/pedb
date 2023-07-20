#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -u debias_adapter.py \
	--model_name_or_path "gpt2" \
	--task_type "causal_lm" \
	--train_file "data/wikipedia-10.txt" \
	--max_seq_length 1024 \
	--line_by_line \
	--bias_type "religion" \
	--cda_mode "partial" \
	--output_dir "checkpoints/religion-gpt2-adapter-rf2" \
	--do_train \
	--per_device_train_batch_size 4 \
	--gradient_accumulation_steps 2 \
	--learning_rate 5e-5 \
	--num_train_epochs 2 \
	--save_strategy "no" \
	--evaluation_strategy "epoch" \
	--seed 42 \
	--down_sample 0.2 \
	--adapter_config "pfeiffer" \
	--adapter_reduction_factor 2 \
	> religion_run_gpt2_adapter_rf2.out 2>&1