import os
import json
from tqdm import tqdm

import torch

import transformers
from transformers import (
	AutoConfig,
	AutoTokenizer,
	AutoModelForMaskedLM,
	AutoModelForCausalLM,
	GPT2LMHeadModel
	)
from datasets import load_dataset

from transformers.adapters.configuration import AdapterConfig
from arguments_adapter import get_args
# from model.utils import get_model
from dataset.language_modeling import get_tokenized_datasets
from bias_bench.benchmark.crows import CrowSPairsRunner
from bias_bench.benchmark.stereoset import StereoSetRunner
from bias_bench.util import generate_experiment_id, _is_generative, _is_self_debias


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_args,data_args,training_args,adapter_args = get_args()

transformers.set_seed(training_args.seed)

# Load config and tokenizer
config_kwargs = {
	"cache_dir": model_args.cache_dir,
	"revision": model_args.model_revision,
	"use_auth_token": True if model_args.use_auth_token else None,
}
if model_args.config_name:
	config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
elif model_args.model_name_or_path:
	config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
else:
	config = CONFIG_MAPPING[model_args.model_type]()
	logger.warning("You are instantiating a new config instance from scratch.")
	if model_args.config_overrides is not None:
		logger.info(f"Overriding config: {model_args.config_overrides}")
		config.update_from_string(model_args.config_overrides)

tokenizer_kwargs = {
	"cache_dir": model_args.cache_dir,
	"use_fast": model_args.use_fast_tokenizer,
	"revision": model_args.model_revision,
	"use_auth_token": True if model_args.use_auth_token else None,
}
if model_args.tokenizer_name:
	tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
elif model_args.model_name_or_path:
	tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
else:
	raise ValueError(
		"You are instantiating a new tokenizer from scratch. This is not supported by this script."
		"You can do it from another script, save it, and load it from here, using --tokenizer_name.")

# Set padding token.
if model_args.task_type=="causal_lm":
	tokenizer.pad_token = tokenizer.eos_token
	config.pad_token_id = config.eos_token_id

# Load model
model_class = AutoModelForMaskedLM if model_args.task_type=="masked_lm" else GPT2LMHeadModel # AutoModelForCausalLM
model = model_class.from_pretrained(
	model_args.model_name_or_path,
	from_tf=bool(".ckpt" in model_args.model_name_or_path),
	config=config,
	cache_dir=model_args.cache_dir,
	revision=model_args.model_revision,
	use_auth_token=True if model_args.use_auth_token else None,
)
# note that for evaluation, `model_args.model_name_or_path` should also be the name in model hub 
model.resize_token_embeddings(len(tokenizer))

# Setup adapters
task_name = model_args.task_type # modified
# check if adapter already exists, otherwise add it
if task_name not in model.config.adapters: # True
	# resolve the adapter config
	adapter_config = AdapterConfig.load(
		adapter_args.adapter_config, # `pfeiffer` if not specify in cmd
		# non_linearity=adapter_args.adapter_non_linearity,
		reduction_factor=adapter_args.adapter_reduction_factor, # None if not specified in cmd
	)
	# load a pre-trained adpater from Hub (or saved path) if specified
	if adapter_args.load_adapter:
		model.load_adapter(
			adapter_args.load_adapter, # what if load_adapter is not consistent with adapter_config? the former is used
			config=adapter_config,
			load_as=task_name, # for line 128
			# with_head=False
		)
	else:
		raise ValueError("should specify a saved adapter via `--load_adapter`")
# Freeze all model weights except of those of this adapter
# model.train_adapter([task_name])
# Set the adapters to be used in every forward pass
model.set_active_adapters(task_name)
model.to(device)
model.eval()

if data_args.dataset_name=='crows':
	runner = CrowSPairsRunner(
		model=model,
		tokenizer=tokenizer,
		input_file=os.path.join('data','crows','crows_pairs_anonymized.csv'),
		bias_type=data_args.bias_type,
		is_generative=_is_generative(model.__class__.__name__),  # Affects model scoring.
		# is_self_debias=_is_self_debias(model.__class__.__name__),
	)
	results = runner() # a number
	print(f"Metric: {results}")
elif data_args.dataset_name=='stereoset':
	runner = StereoSetRunner(
		intrasentence_model=model,
		tokenizer=tokenizer,
		input_file=os.path.join('data','stereoset','test.json'),
		model_name_or_path=model_args.model_name_or_path,
		batch_size=1, # training_args.per_device_eval_batch_size,
		is_generative=_is_generative(model.__class__.__name__),
		# is_self_debias=_is_self_debias(model.__class__.__name__),
		bias_type='race-color' if data_args.bias_type=='race' else data_args.bias_type,
	)
	results = runner() # a nested dict
	# print(f"Metric: {results}")

elif data_args.dataset_name=='wikitext2':
	# To compute (pseudo) perplexity, we consider examples independently. A more reasonable way is to use a 
	# sliding window method to provide some context to the LM.
	raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
	tokenized_dataset = get_tokenized_datasets(data_args,training_args,raw_datasets,tokenizer) # not line_by_line

	print('preparing examples...')
	input_ids = []
	target_ids = []
	num_tokens = []
	for example in tqdm(tokenized_dataset["input_ids"]):
		if model_args.task_type=='masked_lm': # pseudo log-likelihood, pseudo perplexity
			for idx,token_id in enumerate(example):
				if token_id not in [getattr(tokenizer,token+'_id') for token in tokenizer.special_tokens_map]:
					masked_example = example.copy()
					masked_example[idx] = tokenizer.mask_token_id # 103
					input_ids.append(masked_example)
					target_example = [-100]*len(example)
					target_example[idx] = example[idx]
					target_ids.append(target_example)
					num_tokens.append(1)
		elif model_args.task_type=='causal_lm': # perplexity
			input_ids.append(example)
			target_ids.append(example.copy())
			num_tokens.append(len(example)-1)
		else:
			raise ValueError("Invalid argument: task_type")

	print('evaluating...')
	nlls = []
	for start_idx in tqdm(range(0,len(input_ids),training_args.per_device_train_batch_size)):	
		end_idx = min(start_idx+training_args.per_device_train_batch_size,len(input_ids))
		input_ids_tensor = torch.LongTensor(input_ids[start_idx:end_idx]).to(device)
		target_ids_tensor = torch.LongTensor(target_ids[start_idx:end_idx]).to(device)
		with torch.no_grad():
			outputs = model(input_ids_tensor, labels=target_ids_tensor)
			# (loss,logits,past_key_values,hidden_states,attentions,cross_attentions)
			neg_log_likelihood = outputs[0]*sum(num_tokens[start_idx:end_idx]) # (1)
		nlls.append(neg_log_likelihood) # a list of (1) tensors

	results = {f'perplexity_{data_args.max_seq_length}':torch.exp(torch.stack(nlls).sum()/sum(num_tokens)).item()} # perplexity

result_dir = adapter_args.load_adapter # a dir name to debiased model checkpoints
os.makedirs(os.path.join(result_dir,'results'), exist_ok=True)
with open(os.path.join(result_dir,'results',f"{data_args.dataset_name}.json"),"a") as f:
	json.dump(results,f,indent=2)



