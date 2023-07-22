

# bert
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "bert-base-uncased" --prompt_model "none" --task_type "masked_lm" --dataset_name "crows" --bias_type "gender" --seed 42 --output_dir ''
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "bert-base-uncased" --prompt_model "none" --task_type "masked_lm" --dataset_name "stereoset" --bias_type "gender" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "checkpoints/bert-base-uncased"
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "bert-base-uncased" --prompt_model "none" --task_type "masked_lm" --dataset_name "wikitext2" --max_seq_length 480 --bias_type "gender" --seed 42 --output_dir '' --per_device_train_batch_size 32


# bert-fine-tune
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/bert-fine-tune" --prompt_model "none" --task_type "masked_lm" --dataset_name "crows" --bias_type "gender" --seed 42 --output_dir ''
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/bert-fine-tune" --prompt_model "none" --task_type "masked_lm" --dataset_name "stereoset" --bias_type "gender" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "checkpoints/bert-fine-tune"
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/bert-fine-tune" --prompt_model "none" --task_type "masked_lm" --dataset_name "wikitext2" --max_seq_length 480 --bias_type "gender" --seed 42 --output_dir '' --per_device_train_batch_size 32


# bert-prefix-tune-16
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/bert-prefix-tune-16" --prompt_model "prefix_tuning" --pre_seq_len 16 --task_type "masked_lm" --dataset_name "crows" --bias_type "gender" --seed 42 --output_dir ''
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/bert-prefix-tune-16" --prompt_model "prefix_tuning" --pre_seq_len 16 --task_type "masked_lm" --dataset_name "stereoset" --bias_type "gender" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "checkpoints/bert-prefix-tune-16"
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/bert-prefix-tune-16" --prompt_model "prefix_tuning" --pre_seq_len 16 --task_type "masked_lm" --dataset_name "wikitext2" --max_seq_length 480 --bias_type "gender" --seed 42 --output_dir '' --per_device_train_batch_size 32


# bert-prompt-tune-16
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/bert-prompt-tune-16" --prompt_model "prompt_tuning" --pre_seq_len 16 --task_type "masked_lm" --dataset_name "crows" --bias_type "gender" --seed 42 --output_dir ''
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/bert-prompt-tune-16" --prompt_model "prompt_tuning" --pre_seq_len 16 --task_type "masked_lm" --dataset_name "stereoset" --bias_type "gender" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "checkpoints/bert-prompt-tune-16"
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/bert-prompt-tune-16" --prompt_model "prompt_tuning" --pre_seq_len 16 --task_type "masked_lm" --dataset_name "wikitext2" --max_seq_length 480 --bias_type "gender" --seed 42 --output_dir '' --per_device_train_batch_size 32


# bert-adapter-rf48
CUDA_VISIBLE_DEVICES=0 python evaluate_adapter.py --model_name_or_path "bert-base-uncased" --load_adapter "checkpoints/bert-adapter-rf48/masked_lm" --adapter_config "pfeiffer" --adapter_reduction_factor 48 --task_type "masked_lm" --dataset_name "crows" --bias_type "gender" --seed 42 --output_dir ''
CUDA_VISIBLE_DEVICES=0 python evaluate_adapter.py --model_name_or_path "bert-base-uncased" --load_adapter "checkpoints/bert-adapter-rf48/masked_lm" --adapter_config "pfeiffer" --adapter_reduction_factor 48 --task_type "masked_lm" --dataset_name "stereoset" --bias_type "gender" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "checkpoints/bert-adapter-rf48/masked_lm"
CUDA_VISIBLE_DEVICES=0 python evaluate_adapter.py --model_name_or_path "bert-base-uncased" --load_adapter "checkpoints/bert-adapter-rf48/masked_lm" --adapter_config "pfeiffer" --adapter_reduction_factor 48 --task_type "masked_lm" --dataset_name "wikitext2" --max_seq_length 480 --bias_type "gender" --seed 42 --output_dir '' --per_device_train_batch_size 32



# gpt2
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "gpt2" --prompt_model "none" --task_type "causal_lm" --dataset_name "crows" --bias_type "gender" --seed 42 --output_dir ''
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "gpt2" --prompt_model "none" --task_type "causal_lm" --dataset_name "stereoset" --bias_type "gender" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "checkpoints/gpt2"
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "gpt2" --prompt_model "none" --task_type "causal_lm" --dataset_name "wikitext2" --max_seq_length 992 --bias_type "gender" --seed 42 --output_dir '' --per_device_train_batch_size 8


# gpt2-fine-tune
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/gpt2-fine-tune" --prompt_model "none" --task_type "causal_lm" --dataset_name "crows" --bias_type "gender" --seed 42 --output_dir ''
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/gpt2-fine-tune" --prompt_model "none" --task_type "causal_lm" --dataset_name "stereoset" --bias_type "gender" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "checkpoints/gpt2-fine-tune"
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/gpt2-fine-tune" --prompt_model "none" --task_type "causal_lm" --dataset_name "wikitext2" --max_seq_length 992 --bias_type "gender" --seed 42 --output_dir '' --per_device_train_batch_size 4


# gpt2-prefix-tune-16
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/gpt2-prefix-tune-16" --prompt_model "prefix_tuning" --pre_seq_len 16 --task_type "causal_lm" --dataset_name "crows" --bias_type "gender" --seed 42 --output_dir ''
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/gpt2-prefix-tune-16" --prompt_model "prefix_tuning" --pre_seq_len 16 --task_type "causal_lm" --dataset_name "stereoset" --bias_type "gender" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "checkpoints/gpt2-prefix-tune-16"
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/gpt2-prefix-tune-16" --prompt_model "prefix_tuning" --pre_seq_len 16 --task_type "causal_lm" --dataset_name "wikitext2" --max_seq_length 992 --bias_type "gender" --seed 42 --output_dir '' --per_device_train_batch_size 4


# gpt2-prompt-tune-16
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/gpt2-prompt-tune-16" --prompt_model "prompt_tuning" --pre_seq_len 16 --task_type "causal_lm" --dataset_name "crows" --bias_type "gender" --seed 42 --output_dir ''
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/gpt2-prompt-tune-16" --prompt_model "prompt_tuning" --pre_seq_len 16 --task_type "causal_lm" --dataset_name "stereoset" --bias_type "gender" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "checkpoints/gpt2-prompt-tune-16"
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "checkpoints/gpt2-prompt-tune-16" --prompt_model "prompt_tuning" --pre_seq_len 16 --task_type "causal_lm" --dataset_name "wikitext2" --max_seq_length 992 --bias_type "gender" --seed 42 --output_dir '' --per_device_train_batch_size 4


# gpt2-adapter-rf48
CUDA_VISIBLE_DEVICES=0 python evaluate_adapter.py --model_name_or_path "gpt2" --load_adapter "checkpoints/gpt2-adapter-rf48/causal_lm" --adapter_config "pfeiffer" --adapter_reduction_factor 48 --task_type "causal_lm" --dataset_name "crows" --bias_type "gender" --seed 42 --output_dir ''
CUDA_VISIBLE_DEVICES=0 python evaluate_adapter.py --model_name_or_path "gpt2" --load_adapter "checkpoints/gpt2-adapter-rf48/causal_lm" --adapter_config "pfeiffer" --adapter_reduction_factor 48 --task_type "causal_lm" --dataset_name "stereoset" --bias_type "gender" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "checkpoints/gpt2-adapter-rf48/causal_lm"
CUDA_VISIBLE_DEVICES=0 python evaluate_adapter.py --model_name_or_path "gpt2" --load_adapter "checkpoints/gpt2-adapter-rf48/causal_lm" --adapter_config "pfeiffer" --adapter_reduction_factor 48 --task_type "causal_lm" --dataset_name "wikitext2" --max_seq_length 992 --bias_type "gender" --seed 42 --output_dir '' --per_device_train_batch_size 4


# SentDebias
# for bert
cd experiments
python sentence_debias_subspace.py --model "BertModel" --model_name_or_path "bert-base-uncased" --bias_type "gender"
python evaluate.py --model_name_or_path "bert-base-uncased" --prompt_model "SentenceDebiasBertForMaskedLM" --task_type "masked_lm" --dataset_name "crows" --bias_type "gender" --seed 42 --output_dir ''
python evaluate.py --model_name_or_path "bert-base-uncased" --prompt_model "SentenceDebiasBertForMaskedLM" --task_type "masked_lm" --dataset_name "stereoset" --bias_type "gender" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "results/SentenceDebiasBertForMaskedLM_bert-base-uncased"
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "bert-base-uncased" --prompt_model "SentenceDebiasBertForMaskedLM" --task_type "masked_lm" --dataset_name "wikitext2" --max_seq_length 480 --bias_type "gender" --seed 42 --output_dir '' --per_device_train_batch_size 32
# for gpt2
cd experiments
python sentence_debias_subspace.py --model "GPT2Model" --model_name_or_path "gpt2" --bias_type "gender"
python evaluate.py --model_name_or_path "gpt2" --prompt_model "SentenceDebiasGPT2LMHeadModel" --task_type "causal_lm" --dataset_name "crows" --bias_type "gender" --seed 42 --output_dir ''
python evaluate.py --model_name_or_path "gpt2" --prompt_model "SentenceDebiasGPT2LMHeadModel" --task_type "causal_lm" --dataset_name "stereoset" --bias_type "gender" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "results/SentenceDebiasGPT2LMHeadModel_gpt2"
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "gpt2" --prompt_model "SentenceDebiasGPT2LMHeadModel" --task_type "causal_lm" --dataset_name "wikitext2" --max_seq_length 992 --bias_type "gender" --seed 42 --output_dir '' --per_device_train_batch_size 16



# SelfDebias
# for bert
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "bert-base-uncased" --prompt_model "SelfDebiasBertForMaskedLM" --task_type "masked_lm" --dataset_name "crows" --bias_type "gender" --seed 42 --output_dir ''
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "bert-base-uncased" --prompt_model "SelfDebiasBertForMaskedLM" --task_type "masked_lm" --dataset_name "stereoset" --bias_type "gender" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "results/SelfDebiasBertForMaskedLM_bert-base-uncased"
# to evaluate perplexity, you need to modify bias_bench/debias/self_debias/modeling.py according to the comments therein
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "bert-base-uncased" --prompt_model "SelfDebiasBertForMaskedLM" --task_type "masked_lm" --dataset_name "wikitext2" --max_seq_length 480 --bias_type "gender" --seed 42 --output_dir '' --per_device_train_batch_size 32

# for gpt2
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "gpt2" --prompt_model "SelfDebiasGPT2LMHeadModel" --task_type "causal_lm" --dataset_name "crows" --bias_type "gender" --seed 42 --output_dir ''
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "gpt2" --prompt_model "SelfDebiasGPT2LMHeadModel" --task_type "causal_lm" --dataset_name "stereoset" --bias_type "gender" --seed 42 --output_dir ''
python stereoset_evaluation.py --save_dir "results/SelfDebiasGPT2LMHeadModel_gpt2"
# to evaluate perplexity, you need to modify bias_bench/debias/self_debias/modeling.py according to the comments therein
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_name_or_path "gpt2" --prompt_model "SelfDebiasGPT2LMHeadModel" --task_type "causal_lm" --dataset_name "wikitext2" --max_seq_length 992 --bias_type "gender" --seed 42 --output_dir '' --per_device_train_batch_size 8
