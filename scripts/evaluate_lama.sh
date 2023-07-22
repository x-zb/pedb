

## LAMA ##
cd LAMA
export PYTHONPATH=/home/bin/modules:~/pdb/LAMA:~/pdb
# python lama/vocab_intersection.py --output_dir '' (we do not adopt intersectional vocabulary)
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiments.py --model_name_or_path "bert-base-uncased" --prompt_model "none" --task_type "masked_lm" --output_dir '' > bert.out 2>&1 
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiments.py --model_name_or_path "../checkpoints/bert-fine-tune" --prompt_model "none" --task_type "masked_lm" --output_dir '' > bert_fine_tune.out 2>&1 
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiments.py --model_name_or_path "../checkpoints/bert-prefix-tune-16" --prompt_model "prefix_tuning" --pre_seq_len 16 --task_type "masked_lm" --output_dir '' > bert_prefix_tune.out 2>&1 
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiments.py --model_name_or_path "../checkpoints/bert-prompt-tune-16" --prompt_model "prompt_tuning" --pre_seq_len 16 --task_type "masked_lm" --output_dir '' > bert_prompt_tune.out 2>&1 
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiments_adapter.py --model_name_or_path "bert-base-uncased" --load_adapter "../checkpoints/bert-adapter-rf48/masked_lm" --task_type "masked_lm" --output_dir '' > bert_adapter_tune_rf48.out 2>&1 



CUDA_VISIBLE_DEVICES=1 python scripts/run_experiments.py --model_name_or_path "gpt2" --prompt_model "none" --task_type "causal_lm" --output_dir '' > gpt2.out 2>&1 
CUDA_VISIBLE_DEVICES=1 python scripts/run_experiments.py --model_name_or_path "../checkpoints/gpt2-fine-tune" --prompt_model "none" --task_type "causal_lm" --output_dir '' > gpt2_fine_tune.out 2>&1 
CUDA_VISIBLE_DEVICES=1 python scripts/run_experiments.py --model_name_or_path "../checkpoints/gpt2-prefix-tune-16" --prompt_model "prefix_tuning" --pre_seq_len 16 --task_type "causal_lm" --output_dir '' > gpt2_prefix_tune.out 2>&1 
CUDA_VISIBLE_DEVICES=1 python scripts/run_experiments.py --model_name_or_path "../checkpoints/gpt2-prompt-tune-16" --prompt_model "prompt_tuning" --pre_seq_len 16 --task_type "causal_lm" --output_dir '' > gpt2_prompt_tune.out 2>&1 
CUDA_VISIBLE_DEVICES=1 python scripts/run_experiments_adapter.py --model_name_or_path "gpt2" --load_adapter "../checkpoints/gpt2-adapter-rf48/causal_lm" --task_type "causal_lm" --output_dir '' > gpt2_adapter_tune_rf48.out 2>&1 




