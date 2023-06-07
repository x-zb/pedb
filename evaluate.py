import os
import json
from tqdm import tqdm

import torch

import transformers
from transformers import AutoConfig,AutoTokenizer
from datasets import load_dataset

from arguments import get_args
from model.utils import get_model
from dataset.language_modeling import get_tokenized_datasets
from bias_bench.benchmark.crows import CrowSPairsRunner
from bias_bench.benchmark.stereoset import StereoSetRunner
from bias_bench.util import generate_experiment_id, _is_generative, _is_self_debias
from bias_bench.model import models


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_args,data_args,training_args = get_args()

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
bias_bench_models = ["SentenceDebiasBertForMaskedLM","INLPBertForMaskedLM","SelfDebiasBertForMaskedLM",
	"SentenceDebiasGPT2LMHeadModel","INLPGPT2LMHeadModel","SelfDebiasGPT2LMHeadModel"]
if model_args.prompt_model in bias_bench_models:
	debiased_model_to_base_model = {
		"SentenceDebiasBertForMaskedLM":'BertModel',
		"INLPBertForMaskedLM":'BertModel',
		"SelfDebiasBertForMaskedLM":'BertModel',
		"SentenceDebiasGPT2LMHeadModel":'GPT2Model',
		"INLPGPT2LMHeadModel":'GPT2Model',
		"SelfDebiasGPT2LMHeadModel":'GPT2Model'}
	kwargs = {}
	if 'SentenceDebias' in model_args.prompt_model:
		bias_direction = "results/subspace/subspace_m-{}_c-{}_t-{}.pt".format(
			debiased_model_to_base_model[model_args.prompt_model],model_args.model_name_or_path,data_args.bias_type)
		kwargs["bias_direction"] = torch.load(bias_direction)
	if 'INLP' in model_args.prompt_model:
		projection_matrix = "results/projection_matrix/projection_m-{}_c-{}_t-{}_s-0.pt".format(
			debiased_model_to_base_model[model_args.prompt_model],model_args.model_name_or_path,data_args.bias_type)
		kwargs["projection_matrix"] = torch.load(projection_matrix)
	model = getattr(models, model_args.prompt_model)(model_args.model_name_or_path, **kwargs)
	if _is_self_debias(model_args.prompt_model):
		model._model.eval()
		model._model.to(device)
	else:
		model.eval()
		model.to(device)
else:
	if model_args.prefix_tokens is not None:
		model_args.prefix_tokens = tokenizer.encode(model_args.prefix_tokens,add_special_tokens=False)
		print('use real word for initialization, prefix length: {}'.format(len(model_args.prefix_tokens)))
	model = get_model(model_args,config)
	# note that for evaluation, `model_args.model_name_or_path` should be set to the checkpoints saved by debias_xxx.py 
	model.resize_token_embeddings(len(tokenizer))
	model.to(device)
	model.eval()


class_name = model_args.prompt_model if model_args.prompt_model in bias_bench_models else model.__class__.__name__
if data_args.dataset_name=='crows':
	runner = CrowSPairsRunner(
		model=model,
		tokenizer=tokenizer,
		input_file=os.path.join('data','crows','crows_pairs_anonymized.csv'),
		bias_type=data_args.bias_type,
		is_generative=_is_generative(class_name),  # Affects model scoring.
		is_self_debias=_is_self_debias(class_name),
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
		is_generative=_is_generative(class_name),
		is_self_debias=_is_self_debias(class_name),
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
			if _is_self_debias(class_name):
				# Prompts for self-debiasing.
				DEBIASING_PREFIXES = {
					"race-color": "The following text discriminates against people because of their race/color: ",
					"gender": "The following text discriminates against people because of their gender: ",
					"socioeconomic": "The following text discriminates against people because of their socioeconomic status/occupation: ",
					"sexual-orientation": "The following text discriminates against people because of their sexual orientiation: ",
					"religion": "The following text discriminates against people because of their religion: ",
					"age": "The following text discriminates against people because of their age: ",
					"nationality": "The following text discriminates against people because of their nationality: ",
					"disability": "The following text discriminates against people because of their disability: ",
					"physical-appearance": "The following text discriminates against people because of their physical appearance: ",
				}
				loss = 0
				for idx in range(end_idx - start_idx):
					debiasing_prefixes = [DEBIASING_PREFIXES['race-color' if data_args.bias_type=='race' else data_args.bias_type]]
					loss += model.compute_loss_self_debiasing(
						input_ids=input_ids_tensor[idx], # (x).repeat(a,1) or (1,x).repeat(a,1) will both result in (a,x)
						trg_len=data_args.max_seq_length, # for clm
						debiasing_prefixes=debiasing_prefixes,
						decay_constant=50,
						epsilon=0.01,
						debug=False, # for clm
						) if model_args.task_type=='causal_lm' else model.compute_loss_self_debiasing(
						input_ids=input_ids_tensor[idx].unsqueeze(0), # (x).repeat(a,1) or (1,x).repeat(a,1) will both result in (a,x)
						# trg_len=data_args.max_seq_length, # for clm
						labels=target_ids_tensor[idx].unsqueeze(0), # for mlm
						debiasing_prefixes=debiasing_prefixes,
						decay_constant=50,
						epsilon=0.01,
						# debug=False, # for clm
						)
				loss /= (end_idx - start_idx)
			else:
				outputs = model(input_ids_tensor, labels=target_ids_tensor)
				# (loss,logits,past_key_values,hidden_states,attentions,cross_attentions)
				loss = outputs[0]
			neg_log_likelihood = loss*sum(num_tokens[start_idx:end_idx]) # (1)
		nlls.append(neg_log_likelihood) # a list of (1) tensors

	results = {f'perplexity_{data_args.max_seq_length}':torch.exp(torch.stack(nlls).sum()/sum(num_tokens)).item()} # perplexity

result_dir = None
if model_args.prompt_model in bias_bench_models:
	result_dir = os.path.join('results',model_args.prompt_model+'_'+model_args.model_name_or_path)
elif 'checkpoints' in model_args.model_name_or_path: # a dir name to debiased model checkpoints
	result_dir = model_args.model_name_or_path
else: # a name in model hub, e.g. "bert-base-uncased"
	result_dir = os.path.join('checkpoints',model_args.model_name_or_path)
os.makedirs(os.path.join(result_dir,'results'), exist_ok=True)
with open(os.path.join(result_dir,'results',f"{data_args.dataset_name}.json"),"a") as f:
	json.dump(results,f,indent=2)



