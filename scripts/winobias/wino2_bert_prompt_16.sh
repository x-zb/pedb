#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -u task.py \
	--model_name_or_path "bert-base-uncased" \
	--task_type "coref" \
	--prompt_model "prompt_tuning" \
	--pre_seq_len 16 \
	--dataset_config_name "type2" \
	--max_seq_length 496 \
	--validation_split_percentage 5 \
	--output_dir "checkpoints/wino2-bert-prompt-tune-16" \
	--label_names 'answers' \
	--do_train \
	--do_eval \
	--per_device_train_batch_size 16 \
	--learning_rate 5e-1 \
	--num_train_epochs 20 \
	--save_strategy "no" \
	--seed 42 \
	> wino2_bert_prompt_16.out 2>&1 &