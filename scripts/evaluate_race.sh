
# bert 
# note that the results are also stored in checkpoints/bert-base-uncased/results, which is the same as gender and religion; you may manually change the directory name to avoid confliction
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "bert-base-uncased" --prompt_model "none" --task_type "masked_lm" --dataset_name "crows" --bias_type "race" --seed 42 --output_dir ''
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "bert-base-uncased" --prompt_model "none" --task_type "masked_lm" --dataset_name "stereoset" --bias_type "race" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "checkpoints/bert-base-uncased"
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "bert-base-uncased" --prompt_model "none" --task_type "masked_lm" --dataset_name "wikitext2" --max_seq_length 320 --bias_type "race" --seed 42 --output_dir '' --per_device_train_batch_size 32

# race-bert-fine-tune
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/race-bert-fine-tune" --prompt_model "none" --task_type "masked_lm" --dataset_name "crows" --bias_type "race" --seed 42 --output_dir ''
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/race-bert-fine-tune" --prompt_model "none" --task_type "masked_lm" --dataset_name "stereoset" --bias_type "race" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "checkpoints/race-bert-fine-tune"
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/race-bert-fine-tune" --prompt_model "none" --task_type "masked_lm" --dataset_name "wikitext2" --max_seq_length 320 --bias_type "race" --seed 42 --output_dir '' --per_device_train_batch_size 32


# race-bert-prefix-tune-192
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/race-bert-prefix-tune-192" --prompt_model "prefix_tuning" --pre_seq_len 192 --task_type "masked_lm" --dataset_name "crows" --bias_type "race" --seed 42 --output_dir ''
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/race-bert-prefix-tune-192" --prompt_model "prefix_tuning" --pre_seq_len 192 --task_type "masked_lm" --dataset_name "stereoset" --bias_type "race" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "checkpoints/race-bert-prefix-tune-192"
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/race-bert-prefix-tune-192" --prompt_model "prefix_tuning" --pre_seq_len 192 --task_type "masked_lm" --dataset_name "wikitext2" --max_seq_length 320 --bias_type "race" --seed 42 --output_dir '' --per_device_train_batch_size 32


# race-bert-prompt-tune-192
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/race-bert-prompt-tune-192" --prompt_model "prompt_tuning" --pre_seq_len 192 --task_type "masked_lm" --dataset_name "crows" --bias_type "race" --seed 42 --output_dir ''
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/race-bert-prompt-tune-192" --prompt_model "prompt_tuning" --pre_seq_len 192 --task_type "masked_lm" --dataset_name "stereoset" --bias_type "race" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "checkpoints/race-bert-prompt-tune-192"
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/race-bert-prompt-tune-192" --prompt_model "prompt_tuning" --pre_seq_len 192 --task_type "masked_lm" --dataset_name "wikitext2" --max_seq_length 320 --bias_type "race" --seed 42 --output_dir '' --per_device_train_batch_size 32


# race-bert-adapter-rf4
CUDA_VISIBLE_DEVICES=0 python evaluate_adapter.py --model_name_or_path "bert-base-uncased" --load_adapter "checkpoints/race-bert-adapter-rf4/masked_lm" --adapter_config "pfeiffer" --adapter_reduction_factor 4 --task_type "masked_lm" --dataset_name "crows" --bias_type "race" --seed 42 --output_dir ''
CUDA_VISIBLE_DEVICES=0 python evaluate_adapter.py --model_name_or_path "bert-base-uncased" --load_adapter "checkpoints/race-bert-adapter-rf4/masked_lm" --adapter_config "pfeiffer" --adapter_reduction_factor 4 --task_type "masked_lm" --dataset_name "stereoset" --bias_type "race" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "checkpoints/race-bert-adapter-rf4/masked_lm"
CUDA_VISIBLE_DEVICES=0 python evaluate_adapter.py --model_name_or_path "bert-base-uncased" --load_adapter "checkpoints/race-bert-adapter-rf4/masked_lm" --adapter_config "pfeiffer" --adapter_reduction_factor 4 --task_type "masked_lm" --dataset_name "wikitext2" --max_seq_length 320 --bias_type "race" --seed 42 --output_dir '' --per_device_train_batch_size 32



# gpt2
# note that the results are also stored in checkpoints/gpt2/results, which is the same as gender and religion; you may manually change the directory name to avoid confliction
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "gpt2" --prompt_model "none" --task_type "causal_lm" --dataset_name "crows" --bias_type "race" --seed 42 --output_dir ''
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "gpt2" --prompt_model "none" --task_type "causal_lm" --dataset_name "stereoset" --bias_type "race" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "checkpoints/gpt2"
CUDA_VISIBLE_DEVICES=0 python -i evaluate.py --model_name_or_path "gpt2" --prompt_model "none" --task_type "causal_lm" --dataset_name "wikitext2" --max_seq_length 640 --bias_type "race" --seed 42 --output_dir '' --per_device_train_batch_size 4


# race-gpt2-fine-tune
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/race-gpt2-fine-tune" --prompt_model "none" --task_type "causal_lm" --dataset_name "crows" --bias_type "race" --seed 42 --output_dir ''
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/race-gpt2-fine-tune" --prompt_model "none" --task_type "causal_lm" --dataset_name "stereoset" --bias_type "race" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "checkpoints/race-gpt2-fine-tune"
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/race-gpt2-fine-tune" --prompt_model "none" --task_type "causal_lm" --dataset_name "wikitext2" --max_seq_length 640 --bias_type "race" --seed 42 --output_dir '' --per_device_train_batch_size 4


# race-gpt2-prefix-tune-384
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/race-gpt2-prefix-tune-384" --prompt_model "prefix_tuning" --pre_seq_len 384 --task_type "causal_lm" --dataset_name "crows" --bias_type "race" --seed 42 --output_dir ''
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/race-gpt2-prefix-tune-384" --prompt_model "prefix_tuning" --pre_seq_len 384 --task_type "causal_lm" --dataset_name "stereoset" --bias_type "race" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "checkpoints/race-gpt2-prefix-tune-384"
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/race-gpt2-prefix-tune-384" --prompt_model "prefix_tuning" --pre_seq_len 384 --task_type "causal_lm" --dataset_name "wikitext2" --max_seq_length 640 --bias_type "race" --seed 42 --output_dir '' --per_device_train_batch_size 4


# race-gpt2-prompt-tune-384
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/race-gpt2-prompt-tune-384" --prompt_model "prompt_tuning" --pre_seq_len 384 --task_type "causal_lm" --dataset_name "crows" --bias_type "race" --seed 42 --output_dir ''
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/race-gpt2-prompt-tune-384" --prompt_model "prompt_tuning" --pre_seq_len 384 --task_type "causal_lm" --dataset_name "stereoset" --bias_type "race" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "checkpoints/race-gpt2-prompt-tune-384"
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/race-gpt2-prompt-tune-384" --prompt_model "prompt_tuning" --pre_seq_len 384 --task_type "causal_lm" --dataset_name "wikitext2" --max_seq_length 640 --bias_type "race" --seed 42 --output_dir '' --per_device_train_batch_size 4


# race-gpt2-adapter-rf2
CUDA_VISIBLE_DEVICES=0 python evaluate_adapter.py --model_name_or_path "gpt2" --load_adapter "checkpoints/race-gpt2-adapter-gpt2-rf2/causal_lm" --adapter_config "pfeiffer" --adapter_reduction_factor 2 --task_type "causal_lm" --dataset_name "crows" --bias_type "race" --seed 42 --output_dir ''
CUDA_VISIBLE_DEVICES=0 python evaluate_adapter.py --model_name_or_path "gpt2" --load_adapter "checkpoints/race-gpt2-adapter-gpt2-rf2/causal_lm" --adapter_config "pfeiffer" --adapter_reduction_factor 2 --task_type "causal_lm" --dataset_name "stereoset" --bias_type "race" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "checkpoints/race-gpt2-adapter-gpt2-rf2/causal_lm"
CUDA_VISIBLE_DEVICES=0 python evaluate_adapter.py --model_name_or_path "gpt2" --load_adapter "checkpoints/race-gpt2-adapter-gpt2-rf2/causal_lm" --adapter_config "pfeiffer" --adapter_reduction_factor 2 --task_type "causal_lm" --dataset_name "wikitext2" --max_seq_length 640 --bias_type "race" --seed 42 --output_dir '' --per_device_train_batch_size 4



# SentDebias
# for bert
cd experiments
python sentence_debias_subspace.py --model "BertModel" --model_name_or_path "bert-base-uncased" --bias_type "race"
python evaluate.py --model_name_or_path "bert-base-uncased" --prompt_model "SentenceDebiasBertForMaskedLM" --task_type "masked_lm" --dataset_name "crows" --bias_type "race" --seed 42 --output_dir ''
python evaluate.py --model_name_or_path "bert-base-uncased" --prompt_model "SentenceDebiasBertForMaskedLM" --task_type "masked_lm" --dataset_name "stereoset" --bias_type "race" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "results/SentenceDebiasBertForMaskedLM_bert-base-uncased"
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "bert-base-uncased" --prompt_model "SentenceDebiasBertForMaskedLM" --task_type "masked_lm" --dataset_name "wikitext2" --max_seq_length 320 --bias_type "race" --seed 42 --output_dir '' --per_device_train_batch_size 32
# for gpt2
cd experiments
python sentence_debias_subspace.py --model "GPT2Model" --model_name_or_path "gpt2" --bias_type "race"
python evaluate.py --model_name_or_path "gpt2" --prompt_model "SentenceDebiasGPT2LMHeadModel" --task_type "causal_lm" --dataset_name "crows" --bias_type "race" --seed 42 --output_dir ''
python evaluate.py --model_name_or_path "gpt2" --prompt_model "SentenceDebiasGPT2LMHeadModel" --task_type "causal_lm" --dataset_name "stereoset" --bias_type "race" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "results/SentenceDebiasGPT2LMHeadModel_gpt2"
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "gpt2" --prompt_model "SentenceDebiasGPT2LMHeadModel" --task_type "causal_lm" --dataset_name "wikitext2" --max_seq_length 640 --bias_type "race" --seed 42 --output_dir '' --per_device_train_batch_size 8


# SelfDebias
# for bert
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "bert-base-uncased" --prompt_model "SelfDebiasBertForMaskedLM" --task_type "masked_lm" --dataset_name "crows" --bias_type "race" --seed 42 --output_dir ''
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "bert-base-uncased" --prompt_model "SelfDebiasBertForMaskedLM" --task_type "masked_lm" --dataset_name "stereoset" --bias_type "race" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "results/SelfDebiasBertForMaskedLM_bert-base-uncased"
# to evaluate perplexity, you need to modify bias_bench/debias/self_debias/modeling.py according to the comments therein
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "bert-base-uncased" --prompt_model "SelfDebiasBertForMaskedLM" --task_type "masked_lm" --dataset_name "wikitext2" --max_seq_length 320 --bias_type "race" --seed 42 --output_dir '' --per_device_train_batch_size 32

# for gpt2
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "gpt2" --prompt_model "SelfDebiasGPT2LMHeadModel" --task_type "causal_lm" --dataset_name "crows" --bias_type "race" --seed 42 --output_dir ''
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "gpt2" --prompt_model "SelfDebiasGPT2LMHeadModel" --task_type "causal_lm" --dataset_name "stereoset" --bias_type "race" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "results/SelfDebiasGPT2LMHeadModel_gpt2"
# to evaluate perplexity, you need to modify bias_bench/debias/self_debias/modeling.py according to the comments therein
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "gpt2" --prompt_model "SelfDebiasGPT2LMHeadModel" --task_type "causal_lm" --dataset_name "wikitext2" --max_seq_length 640 --bias_type "race" --seed 42 --output_dir '' --per_device_train_batch_size 8


