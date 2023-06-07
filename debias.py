import sys
import os
import math
import logging
import random
from tqdm import tqdm
import pickle

import torch

import transformers
from transformers import (
	AutoConfig,
	AutoTokenizer,
	# BertForMaskedLM,
	GPT2LMHeadModel,
	DataCollatorForLanguageModeling,
	Trainer,
	set_seed
	)
from transformers.trainer_utils import get_last_checkpoint
import datasets
from datasets import load_from_disk

from arguments import get_args
from dataset.language_modeling import get_raw_datasets,get_tokenized_datasets,counterfactual_data_augmentation
from model.utils import get_model,DataCollatorForDebiasingCLM

logger = logging.getLogger(__name__)

model_args,data_args,training_args = get_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setup logging
logging.basicConfig(
	format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
	datefmt="%m/%d/%Y %H:%M:%S",
	handlers=[logging.StreamHandler(sys.stdout)],
)
log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()
# Log on each process the small summary:
logger.warning(
	f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
	+ f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
# Set the verbosity to info of the Transformers logger (on main process only):
logger.info(f"Training/evaluation parameters {training_args}")

# Set seed before initializing model.
set_seed(training_args.seed) 
# Helper function for reproducible behavior to set the seed in random, numpy, torch and/or tf (if installed).

# Load config and tokenizer
# Distributed training:
# The .from_pretrained methods guarantee that only one local process can concurrently
# download model & vocab.
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
if model_args.prefix_tokens is not None:
	model_args.prefix_tokens = tokenizer.encode(model_args.prefix_tokens,add_special_tokens=False)
	print('use real word for initialization, prefix length: {}'.format(len(model_args.prefix_tokens)))
model = get_model(model_args,config)
model.resize_token_embeddings(len(tokenizer)) 
# Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.
# Increasing `new_num_tokens` will add newly initialized vectors at the end. Reducing it will remove vectors from the end.

# Load datasets
cda_dataset_dir = "data/wikipedia-10-"+data_args.bias_type+'-'+str(data_args.max_seq_length)+("-linebyline" if data_args.line_by_line else "-block")
# cda_dataset_dir = "data/openwebtext-"+data_args.bias_type+'-'+str(data_args.max_seq_length)+("-linebyline" if data_args.line_by_line else "-block")
if os.path.exists(cda_dataset_dir):
	print('Using saved cda dataset from: '+cda_dataset_dir)
	cda_datasets = load_from_disk(cda_dataset_dir)
else:
	raw_datasets = get_raw_datasets(model_args,data_args)
	tokenized_datasets = get_tokenized_datasets(data_args,training_args,raw_datasets,tokenizer)
	cda_datasets = counterfactual_data_augmentation(data_args,tokenized_datasets,tokenizer)
	cda_datasets.save_to_disk(cda_dataset_dir)

if training_args.do_train:
	if "train" not in cda_datasets:
		raise ValueError("--do_train requires a train dataset")
	train_dataset = cda_datasets["train"]
	if data_args.max_train_samples is not None:
		train_dataset = train_dataset.select(range(data_args.max_train_samples))

if training_args.do_eval:
	if "validation" not in cda_datasets:
		raise ValueError("--do_eval requires a validation dataset")
	eval_dataset = cda_datasets["validation"]
	if data_args.max_eval_samples is not None:
		eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

if data_args.down_sample>0: 
# down sample the train_dataset and guarantee that (original,augmented) are selected togother
	n_group = 2 if data_args.bias_type=='gender' else 3 # 6
	if data_args.down_sample_strategy=='random':
		sampled_indices = []
		sample_size = int(int(len(train_dataset)/n_group)*data_args.down_sample)
		random.seed(42) # guarantee that the same subset is sampled for different seeds
		for idx in random.sample(range(int(len(train_dataset)/n_group)),sample_size):
			for i in range(n_group):
				sampled_indices.append(idx*n_group+i)
			# sampled_indices.append(idx*2)
			# sampled_indices.append(idx*2+1) # guarantee that (original,augmented) are selected togother
		train_dataset = train_dataset.select(sampled_indices)
		print(f'sampled train_dataset size: {len(train_dataset)} ({data_args.down_sample*100}% of the augmented dataset)')
		random.seed(training_args.seed)
	else:
		assert data_args.down_sample_strategy=='gpt2'
		sorted_indices_file = "data/gpt2-sorted-wikipedia-10-"+data_args.bias_type+'-'+str(data_args.max_seq_length)+("-linebyline" if data_args.line_by_line else "-block")+'.bin'
		if os.path.exists(sorted_indices_file):
			print('Using saved indices from: '+sorted_indices_file)
			with open(sorted_indices_file,'rb') as f:
				sorted_indices = pickle.load(f)
		else:
			filter_model = GPT2LMHeadModel.from_pretrained(model_args.model_name_or_path,cache_dir=model_args.cache_dir)
			filter_model.eval()
			filter_model.to(device)
			filter_nlls = []
			print('Evaluating nlls for gpt2-based down sample')
			for idx in tqdm(range(int(len(train_dataset)/2))):
				filter_nll = filter_model(
					input_ids=torch.LongTensor(train_dataset[idx*2]['input_ids']).to(device),
					attention_mask=torch.LongTensor(train_dataset[idx*2]['attention_mask']).to(device),
					labels=torch.LongTensor(train_dataset[idx*2]['input_ids'].copy()).to(device)
				)[0] # (1)
				filter_nlls.append(filter_nll.item())
			sorted_indices = sorted(range(len(filter_nlls)),key=lambda k: filter_nlls[k],reverse=False) # in ascending orders
			with open(sorted_indices_file,'wb') as f:
				pickle.dump(sorted_indices,f)
			with open("data/gpt2-filter_nlls-wikipedia-10-"+data_args.bias_type+'-'+str(data_args.max_seq_length)+("-linebyline" if data_args.line_by_line else "-block")+'.bin','wb') as f:
				pickle.dump(filter_nlls,f)
		
		sample_size = int(int(len(train_dataset)/2)*data_args.down_sample)
		sampled_indices = sum([[idx*2,idx*2+1] for idx in sorted_indices[:sample_size]],[])
		train_dataset = train_dataset.select(sampled_indices)
		print(f'sampled train_dataset size: {len(train_dataset)} ({data_args.down_sample*100}% of the augmented dataset)')
		

# Data collator
# This one will take care of randomly masking the tokens.
# pad_to_multiple_of_8 = (
# 	data_args.line_by_line
# 	and training_args.fp16 # False
# 	and not data_args.pad_to_max_length # not False
# )
if model_args.task_type=="masked_lm":
	data_collator = DataCollatorForLanguageModeling(
		tokenizer=tokenizer,
		mlm=True,
		mlm_probability=data_args.mlm_probability,
		# pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
		return_tensors='pt'
	)
elif model_args.task_type=="causal_lm":
	data_collator = DataCollatorForLanguageModeling(
		tokenizer=tokenizer,
		mlm=False,
		# pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
		return_tensors='pt'
	)


# Initialize our Trainer
trainer = Trainer(
	model=model,
	args=training_args,
	data_collator=data_collator,
	train_dataset=train_dataset if training_args.do_train else None,
	eval_dataset=eval_dataset if training_args.do_eval else None, # training_args.do_eval is True
	tokenizer=tokenizer,
)

# Detecting last checkpoint.
last_checkpoint = None
if (
	os.path.isdir(training_args.output_dir)
	and training_args.do_train
	and not training_args.overwrite_output_dir
):
	last_checkpoint = get_last_checkpoint(training_args.output_dir)
	if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
		raise ValueError(
		# logger.info(
			f"Output directory ({training_args.output_dir}) already exists and is not empty. "
			"Use --overwrite_output_dir to overcome."
		)
	elif (
		last_checkpoint is not None and training_args.resume_from_checkpoint is None
	):
		logger.info(
			f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
			"the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
		)

# Training
if training_args.do_train:
	checkpoint = None
	if training_args.resume_from_checkpoint is not None:
		checkpoint = training_args.resume_from_checkpoint
	elif last_checkpoint is not None:
		checkpoint = last_checkpoint
	train_result = trainer.train(resume_from_checkpoint=checkpoint) # TrainOutput object
	trainer.save_model()  # Saves the tokenizer too for easy upload
	# save to training_args.output_dir
	metrics = train_result.metrics

	max_train_samples = (
		data_args.max_train_samples
		if data_args.max_train_samples is not None
		else len(train_dataset)
	)
	metrics["train_samples"] = min(max_train_samples, len(train_dataset))

	trainer.log_metrics("train", metrics)
	trainer.save_metrics("train", metrics)
	trainer.save_state()

# Evaluation
if training_args.do_eval: # Will be set to True if evaluation_strategy is different from "no". 
	
	logger.info("*** Evaluate ***")

	metrics = trainer.evaluate() # {'eval_loss':loss}

	max_eval_samples = (
		data_args.max_eval_samples
		if data_args.max_eval_samples is not None
		else len(eval_dataset)
	)
	metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
	try:
		perplexity = math.exp(metrics["eval_loss"])
	except OverflowError:
		perplexity = float("inf")
	metrics["perplexity"] = perplexity

	trainer.log_metrics("eval", metrics)
	trainer.save_metrics("eval", metrics)
