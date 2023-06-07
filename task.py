import sys
import os
import math
import logging
import random

import numpy as np

import transformers
from transformers import (
	AutoConfig,
	AutoTokenizer,
	# DataCollatorForLanguageModeling,
	Trainer,
	set_seed
	)
from transformers.trainer_utils import get_last_checkpoint
import datasets
from datasets import load_from_disk,load_metric

from arguments import get_args
from dataset.multiple_choice import get_winobias,DataCollatorForMultipleChoiceAndTokenClassification
from model.utils import get_model

logger = logging.getLogger(__name__)

model_args,data_args,training_args = get_args()

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
if model_args.task_type=="causal_lm" or 'gpt2' in model_args.model_name_or_path:
	tokenizer.pad_token = tokenizer.eos_token
	config.pad_token_id = config.eos_token_id


# Load model
if model_args.prefix_tokens is not None:
	model_args.prefix_tokens = tokenizer.encode(model_args.prefix_tokens,add_special_tokens=False)['input_ids']
model = get_model(model_args,config)
model.resize_token_embeddings(len(tokenizer)) 
# Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.
# Increasing `new_num_tokens` will add newly initialized vectors at the end. Reducing it will remove vectors from the end.

# Load datasets
if model_args.task_type=='coref':
	tokenized_datasets = get_winobias(model_args,data_args,training_args,config,tokenizer)
elif model_args.task_type=='nli':
	pass



if training_args.do_train:
	if "train" not in tokenized_datasets:
		raise ValueError("--do_train requires a train dataset")
	train_dataset = tokenized_datasets["train"]
	if data_args.max_train_samples is not None:
		train_dataset = train_dataset.select(range(data_args.max_train_samples))

# if training_args.do_eval:
	if "validation" not in tokenized_datasets:
		raise ValueError("--do_eval requires a validation dataset")
	eval_dataset = tokenized_datasets["validation"]
	if data_args.max_eval_samples is not None:
		eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

# Data collator
if model_args.task_type=="coref":
	data_collator = DataCollatorForMultipleChoiceAndTokenClassification(
		tokenizer=tokenizer,
		# pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
	)
	winobias_metric = load_metric('f1')
	def compute_metrics(eval_predictions):
		# assume the models only output (loss,logits), then 
		# predictions= numpy typed logits~(bsz,n_choices), 
		# labels~(bsz) is extracted from the inputs according to trainer.label_names (specified by TrainingArguments.label_names 
		# or the input arguments of model.forward whose name contains 'label')
		predictions, labels = eval_predictions
		if isinstance(predictions, tuple):
			# Depending on the model and config, logits may contain extra tensors,
			# like past_key_values, but logits always come first
			predictions = predictions[0]
		predictions = np.argmin(predictions, axis=1) # (bsz)
		results = winobias_metric.compute(predictions=predictions, references=labels)
		return results # {'f1':float}
elif model_args.task_type=="nli":
	data_collator = DataCollatorForLanguageModeling(
		tokenizer=tokenizer,
		# pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
	)

# Training
if training_args.do_train:

	trainer = Trainer(
		model=model,
		args=training_args,
		data_collator=data_collator,
		train_dataset=train_dataset, # if training_args.do_train else None,
		eval_dataset=eval_dataset, # if training_args.do_eval else None,
		tokenizer=tokenizer,
		compute_metrics=compute_metrics,
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


	checkpoint = None
	if training_args.resume_from_checkpoint is not None:
		checkpoint = training_args.resume_from_checkpoint
	elif last_checkpoint is not None:
		checkpoint = last_checkpoint
	train_result = trainer.train(resume_from_checkpoint=checkpoint)
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
# if training_args.do_eval:
	logger.info("*** Evaluate ***")

	metrics = trainer.evaluate()

	max_eval_samples = (
		data_args.max_eval_samples
		if data_args.max_eval_samples is not None
		else len(eval_dataset)
	)
	metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

	trainer.log_metrics("eval", metrics)
	trainer.save_metrics("eval", metrics)


if training_args.do_eval:
	logger.info("*** Test ***")
	test_metrics = {}
	tester_pro = Trainer(
		model=model,
		args=training_args,
		data_collator=data_collator,
		train_dataset=None,
		eval_dataset=tokenized_datasets["test_pro"],
		tokenizer=tokenizer,
		compute_metrics=compute_metrics,
	)
	test_metrics['f1_pro'] = tester_pro.evaluate()['eval_f1']

	tester_anti = Trainer(
		model=model,
		args=training_args,
		data_collator=data_collator,
		train_dataset=None,
		eval_dataset=tokenized_datasets["test_anti"],
		tokenizer=tokenizer,
		compute_metrics=compute_metrics,
	)
	test_metrics['f1_anti'] = tester_anti.evaluate()['eval_f1']

	test_metrics['diff'] = test_metrics['f1_pro']-test_metrics['f1_anti']
	tester_anti.log_metrics("test", test_metrics)
	tester_anti.save_metrics("test", test_metrics)