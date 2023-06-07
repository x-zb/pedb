from enum import Enum

from collections.abc import Mapping
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from transformers import (
	AutoConfig,
	BertForPreTraining,
	AutoModelForMaskedLM,
	AutoModelForSequenceClassification,
	DataCollatorForLanguageModeling
	)

from model.bert_mlm import (
	BertForMaskedLM,
	BertPrefixTuningForMaskedLM,
	BertPromptTuningForMaskedLM,
	BertPromptingForMaskedLM
	)

from model.gpt2_clm import (
	GPT2LMHeadModel,
	GPT2PrefixTuningLMHeadModel,
	GPT2PromptTuningLMHeadModel,
	GPT2PromptingLMHeadModel
	)

from model.bert_mc_mlm import (
	BertForMultipleChoiceMaskedLM,
	BertPrefixTuningForMultipleChoiceMaskedLM,
	BertPromptTuningForMultipleChoiceMaskedLM,
	BertPromptingForMultipleChoiceMaskedLM
	)

from model.gpt2_mc_clm import (
	GPT2MultipleChoiceLMHeadModel,
	GPT2PrefixTuningMultipleChoiceLMHeadModel,
	GPT2PromptTuningMultipleChoiceLMHeadModel,
	GPT2PromptingMultipleChoiceLMHeadModel
	)


class TaskType(Enum):
	MASKED_LM = 1,
	CAUSAL_LM = 2,
	MULTIPLE_CHOICE =3,
	SEQUENCE_CLASSIFICATION = 4

PREFIXTUNE_MODELS = {
	'bert':{
		TaskType.MASKED_LM: BertPrefixTuningForMaskedLM,
		TaskType.CAUSAL_LM: None,
		TaskType.MULTIPLE_CHOICE: BertPrefixTuningForMultipleChoiceMaskedLM,
	},
	'gpt2':{
		TaskType.MASKED_LM: None,
		TaskType.CAUSAL_LM: GPT2PrefixTuningLMHeadModel,
		TaskType.MULTIPLE_CHOICE: GPT2PrefixTuningMultipleChoiceLMHeadModel,
	}
}

PROMPTTUNE_MODELS = {
	'bert':{
		TaskType.MASKED_LM: BertPromptTuningForMaskedLM,
		TaskType.CAUSAL_LM: None,
		TaskType.MULTIPLE_CHOICE: BertPromptTuningForMultipleChoiceMaskedLM,
	},
	'gpt2':{
		TaskType.MASKED_LM: None,
		TaskType.CAUSAL_LM: GPT2PromptTuningLMHeadModel,
		TaskType.MULTIPLE_CHOICE: GPT2PromptTuningMultipleChoiceLMHeadModel,
	}
}

PROMPTING_MODELS = {
	'bert':{
		TaskType.MASKED_LM: BertPromptingForMaskedLM,
		TaskType.CAUSAL_LM: None,
		TaskType.MULTIPLE_CHOICE: BertPromptingForMultipleChoiceMaskedLM,
	},
	'gpt2':{
		TaskType.MASKED_LM: None,
		TaskType.CAUSAL_LM: GPT2PromptingLMHeadModel,
		TaskType.MULTIPLE_CHOICE: GPT2PromptingMultipleChoiceLMHeadModel,
	}
}


BASE_MODELS = {
	'bert':{
		TaskType.MASKED_LM: BertForMaskedLM,
		TaskType.CAUSAL_LM: None,
		TaskType.MULTIPLE_CHOICE: BertForMultipleChoiceMaskedLM,
	},
	'gpt2':{
		TaskType.MASKED_LM: None,
		TaskType.CAUSAL_LM: GPT2LMHeadModel,
		TaskType.MULTIPLE_CHOICE: GPT2MultipleChoiceLMHeadModel,
	}
}


def get_model(model_args, config: AutoConfig, fix_bert: bool = False):
	
	str2enum = {"masked_lm":TaskType.MASKED_LM,"causal_lm":TaskType.CAUSAL_LM,
		"coref":TaskType.MULTIPLE_CHOICE,"nli":TaskType.SEQUENCE_CLASSIFICATION}
	task_type = str2enum[model_args.task_type]

	if model_args.prompt_model=='prefix_tuning':
		config.hidden_dropout_prob = model_args.hidden_dropout_prob
		config.pre_seq_len = model_args.pre_seq_len
		config.prefix_projection = model_args.prefix_projection
		config.prefix_hidden_size = model_args.prefix_hidden_size
		config.prefix_tokens = model_args.prefix_tokens
		model_class = PREFIXTUNE_MODELS[config.model_type][task_type]

	elif model_args.prompt_model=='prompt_tuning':
		config.pre_seq_len = model_args.pre_seq_len
		config.prefix_tokens = model_args.prefix_tokens
		model_class = PROMPTTUNE_MODELS[config.model_type][task_type]
		
	elif model_args.prompt_model=='prompting':
		config.prefix_tokens = model_args.prefix_tokens
		model_class = PROMPTING_MODELS[config.model_type][task_type]
	else:
		assert model_args.prompt_model=='none'
		model_class = BASE_MODELS[config.model_type][task_type]

	if model_args.model_name_or_path:
		
		if 'zari-bert' in model_args.model_name_or_path:
			temp_model = BertForPreTraining.from_pretrained(
				model_args.model_name_or_path,
				from_tf=True,
				config=config)
			mlm_config = AutoConfig.from_pretrained("bert-large-uncased",cache_dir=model_args.cache_dir)
			model = AutoModelForMaskedLM.from_config(mlm_config)
			model.bert = temp_model.bert
			model.cls.predictions = temp_model.cls.predictions
		else:
			model = model_class.from_pretrained(
				model_args.model_name_or_path,
				# from_tf=bool(".ckpt" in model_args.model_name_or_path),
				# from_tf=model_args.from_tf,
				from_tf=False,
				config=config,
				cache_dir=model_args.cache_dir,
				revision=model_args.model_revision,
				use_auth_token=True if model_args.use_auth_token else None,
			)
	else:
		logger.info("Training new model from scratch")
		model = model_class.from_config(config)
	
	# parameter counting should be done after from_pretrained(), rather than in cls.__init__()
	# tie_weights() are called in from_pretrained() after cls.__init__() to tie input and output embeddings for bert and gpt2
	tunable_param = 0
	frozen_param = 0
	for name,param in model.named_parameters():
		if param.requires_grad:
			tunable_param += param.numel()
		else:
			frozen_param += param.numel()
	print('tunable_param is {}, frozen_param is {}'.format(tunable_param,frozen_param))
	
	return model

class DataCollatorForDebiasingCLM(DataCollatorForLanguageModeling):
	'''we only use `return_tensors == "pt"` in our code'''
	def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
		# Handle dict or lists with proper padding and conversion to tensor.
		if isinstance(examples[0], Mapping):
			batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
		else:
			batch = {
				"input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
			}

		# If special token mask has been preprocessed, pop it from the dict.
		special_tokens_mask = batch.pop("special_tokens_mask", None)
		if self.mlm:
			raise ValueError('DataCollatorForDebiasingCLM is for causal language modeling')
			# batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
			#     batch["input_ids"], special_tokens_mask=special_tokens_mask
			# )
		else:
			batch["input_ids"], labels = self.torch_mask_tokens(
				batch["input_ids"], special_tokens_mask=special_tokens_mask
			)
			# labels = batch["input_ids"].clone()
			if self.tokenizer.pad_token_id is not None:
				labels[labels == self.tokenizer.pad_token_id] = -100
			batch["labels"] = labels
		return batch

	def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
		"""
		Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
		Modified: only prepare labels, not inputs
		"""
		import torch

		labels = inputs.clone()
		# We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
		probability_matrix = torch.full(labels.shape, self.mlm_probability)
		if special_tokens_mask is None:
			special_tokens_mask = [
				self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
			]
			special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
		else:
			special_tokens_mask = special_tokens_mask.bool()

		probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
		masked_indices = torch.bernoulli(probability_matrix).bool()
		labels[~masked_indices] = -100  # We only compute loss on masked tokens

		return inputs, labels
