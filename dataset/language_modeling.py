import os
import logging
import random
import json
from functools import partial
import itertools

from datasets import load_dataset,DatasetDict

logger = logging.getLogger(__name__)

def get_raw_datasets(model_args,data_args):
	
	if data_args.dataset_name is not None:
		# Downloading and loading a dataset from the hub.
		if data_args.dataset_name=='openwebtext':
			raw_datasets = DatasetDict()
			raw_datasets['train'] = load_dataset(
				data_args.dataset_name,
				data_args.dataset_config_name,
				split="train[19%:]", # 0.2*0.05=0.01
				cache_dir=model_args.cache_dir,
			)
			raw_datasets['validation'] = load_dataset(
				data_args.dataset_name,
				data_args.dataset_config_name,
				split="train[19%:20%]",
				cache_dir=model_args.cache_dir,
			)
		else:
			raw_datasets = load_dataset(
				data_args.dataset_name,
				data_args.dataset_config_name,
				cache_dir=model_args.cache_dir,
			)
			if "validation" not in raw_datasets.keys():
				raw_datasets["validation"] = load_dataset(
					data_args.dataset_name,
					data_args.dataset_config_name,
					split=f"train[:{data_args.validation_split_percentage}%]",
					cache_dir=model_args.cache_dir,
				)
				raw_datasets["train"] = load_dataset(
					data_args.dataset_name,
					data_args.dataset_config_name,
					split=f"train[{data_args.validation_split_percentage}%:]",
					cache_dir=model_args.cache_dir,
				)
	else:
		data_files = {}
		if data_args.train_file is not None:
			data_files["train"] = data_args.train_file
			extension = data_args.train_file.split(".")[-1]
		if data_args.validation_file is not None:
			data_files["validation"] = data_args.validation_file
			extension = data_args.validation_file.split(".")[-1]
		if extension == "txt":
			extension = "text"
		raw_datasets = load_dataset(
			extension, data_files=data_files, cache_dir=model_args.cache_dir
		)

		# If no validation data is there, validation_split_percentage will be used to divide the dataset.
		if "validation" not in raw_datasets.keys():
			raw_datasets["validation"] = load_dataset(
				extension,
				data_files=data_files,
				split=f"train[:{data_args.validation_split_percentage}%]",
				cache_dir=model_args.cache_dir,
			)
			raw_datasets["train"] = load_dataset(
				extension,
				data_files=data_files,
				split=f"train[{data_args.validation_split_percentage}%:]",
				cache_dir=model_args.cache_dir,
			)
	return raw_datasets


def get_tokenized_datasets(data_args,training_args,raw_datasets,tokenizer):

	# Preprocessing the datasets.
	# First we tokenize all the texts.
	if training_args.do_train:
		column_names = raw_datasets["train"].column_names
	elif training_args.do_eval:
		column_names = raw_datasets["validation"].column_names
	else:
		column_names = raw_datasets.column_names
	text_column_name = "text" if "text" in column_names else column_names[0]

	if data_args.max_seq_length is None:
		max_seq_length = tokenizer.model_max_length
		if max_seq_length > 1024:
			logger.warning(
				f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
				"Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
			)
			max_seq_length = 1024
	else:
		if data_args.max_seq_length > tokenizer.model_max_length:
			logger.warning(
				f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
				f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
			)
		max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

	if data_args.line_by_line:
		# When using line_by_line, we just tokenize each non-empty line.
		padding = "max_length" if data_args.pad_to_max_length else False 
		# Pad to a maximum length specified with the argument max_length or to the maximum acceptable 
		# input length for the model if that argument is not provided.

		def tokenize_function(examples):
			# Remove empty lines
			examples[text_column_name] = [
				line
				for line in examples[text_column_name]
				if len(line) > 0 and not line.isspace()
			]
			return tokenizer(
				examples[text_column_name],
				padding=padding, # False
				truncation=True, # truncate to max_length
				max_length=max_seq_length, # after special tokens are added
				# We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
				# receives the `special_tokens_mask`.
				return_special_tokens_mask=True,
			)

		with training_args.main_process_first(desc="dataset map tokenization"):
			tokenized_datasets = raw_datasets.map(
				tokenize_function,
				batched=True,
				num_proc=data_args.preprocessing_num_workers,
				remove_columns=[text_column_name],
				load_from_cache_file= False, # not data_args.overwrite_cache,
				desc="Running tokenizer on dataset line_by_line",
			)
	else:
		# Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
		# We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
		# efficient when it receives the `special_tokens_mask`.
		def tokenize_function(examples):
			# if data_args.dataset_name=='openwebtext': # if 'gpt2' in tokenizer.name_or_path:
				# print('adding <|endoftext|>')
				# examples[text_column_name] = [example_text+"<|endoftext|>" for example_text in examples[text_column_name]]
			return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

		with training_args.main_process_first(desc="dataset map tokenization"):
			tokenized_datasets = raw_datasets.map(
				tokenize_function,
				batched=True,
				num_proc=data_args.preprocessing_num_workers,
				remove_columns=column_names, 
				# remove all the original columns, so the returned dataset will only contain 
				# the columns returned by the tokenizer
				load_from_cache_file=False, # not data_args.overwrite_cache,
				desc="Running tokenizer on every text in dataset",
			)
		# the warning "Token indices sequence length is longer than the maximum specified sequence length 
		# for this model (525>512)..." is ok, since we will truncate the sequence using group_texts() below

		# Main data processing function that will concatenate all texts from our dataset and generate chunks of
		# max_seq_length.
		def group_texts(examples):
			# Concatenate all texts.
			concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()} 
			
			total_length = len(concatenated_examples[list(examples.keys())[0]])
			# We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
			# customize this part to your needs.
			if total_length >= max_seq_length:
				total_length = (total_length // max_seq_length) * max_seq_length
			
			# Split by chunks of max_len.
			result = {k:[t[i:i+max_seq_length] for i in range(0,total_length,max_seq_length)]for k,t in concatenated_examples.items()}


			return result # {'input_ids':[[ids of 512],[ids of 512],...,[ids of 512]],...}

		# Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
		# remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
		# might be slower to preprocess.
		#
		# To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
		# https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

		with training_args.main_process_first(desc="grouping texts together"):
			tokenized_datasets = tokenized_datasets.map(
				group_texts,
				batched=True,
				num_proc=data_args.preprocessing_num_workers,
				load_from_cache_file=False, # not data_args.overwrite_cache,
				desc=f"Grouping texts in chunks of {max_seq_length}",
			)
	return tokenized_datasets


def counterfactual_data_augmentation(data_args,tokenized_datasets,tokenizer):
	
	if data_args.max_seq_length is None:
		max_seq_length = tokenizer.model_max_length
		if max_seq_length > 1024:
			logger.warning(
				f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
				"Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
			)
			max_seq_length = 1024
	else:
		if data_args.max_seq_length > tokenizer.model_max_length:
			logger.warning(
				f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
				f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
			)
		max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

	def _create_bias_attribute_words(attribute_file, bias_type):
		"""Creates list of bias attribute words (e.g., he/she).

		Args:
			attribute_file: Path to the file containing the bias attribute words.
			bias_type: Type of bias attribute words to load. Must be one of
				["gender", "race", "religion"].

		Notes:
			* We combine each bias attribute word with several punctuation marks.
			  The current set of words is *not* exhaustive, however, it should
			  cover most occurances.
		"""
		with open(attribute_file, "r") as f:
			bias_attribute_words = json.load(f)[bias_type]

		result = bias_attribute_words[:]
		for punctuation in [".", ",", "?", "!", ";", ":"]:
			for words in bias_attribute_words:
				augmented_words = [word + punctuation for word in words]
				result.append(augmented_words)
		return result

	def gender_counterfactual_augmentation(examples,bias_attribute_words,cda_mode):
		"""Applies gender counterfactual data augmentation to a batch of examples.

		Notes:
			* We apply CDA after the examples have potentially been grouped.
			* This implementation can be made more efficient by operating on
			  token IDs as opposed to text. We currently decode each example
			  as it is simpler.
		"""
		outputs = []
		for input_ids in examples["input_ids"]: # examples["input_ids"] is List[List[int]], input_ids is List[int]
			# For simplicity, decode each example. It is easier to apply augmentation
			# on text as opposed to token IDs.
			sentence = tokenizer.decode(input_ids)
			words = sentence.split()  # Tokenize based on whitespace.
			augmented_sentence = words[:] # this will copy the list so that change to `augmented_sentence` will not change `words`

			augmented = False
			for position, word in enumerate(words):
				for male_word, female_word in bias_attribute_words:
					if male_word == word:
						augmented = True
						augmented_sentence[position] = female_word

					if female_word == word:
						augmented = True
						augmented_sentence[position] = male_word

			# exclude the sentence pair if cda_mode=='partial' and augmented=False
			if cda_mode=='complete' or augmented:
				outputs.append(sentence)
				augmented_sentence = " ".join(augmented_sentence)
				outputs.append(augmented_sentence)
				
		# There are potentially no counterfactual examples.
		if not outputs:
			return {"input_ids": [], "attention_mask": []}

		return tokenizer(
			outputs,
			return_special_tokens_mask=True,
			add_special_tokens=False,  # Special tokens are already added.
			truncation=True,
			padding=False, # True
			max_length=max_seq_length,
		) 
		# if the returned dict has the same keys as the input examples, the same key's values in the dataset
		# will be updated; so no need to remove the old instances


	def ternary_counterfactual_augmentation(examples,bias_attribute_words,cda_mode):
		"""Applies racial/religious counterfactual data augmentation to a batch of
		examples.

		Notes:
			* We apply CDA after the examples have potentially been grouped.
			* This implementation can be made more efficient by operating on
			  token IDs as opposed to text. We currently decode each example
			  as it is simpler.
		"""

		n_groups = {1:0,2:0,3:0}
		outputs = []
		for input_ids in examples["input_ids"]:
			# For simplicity, decode each example. It is easier to apply augmentation
			# on text as opposed to token IDs.
			sentence = tokenizer.decode(input_ids)
			words = sentence.split()  # Tokenize based on whitespace.
			augmented_sentence_1 = words[:]
			augmented_sentence_2 = words[:]


			augmented = False
			substitution_info = {0:[],1:[],2:[]}
			for position, word in enumerate(words):
				for i,augmentation_words in enumerate(bias_attribute_words):
				
					r1_word, r2_word, r3_word = augmentation_words

					if r1_word == word:
						augmented = True
						substitution_info[0].append((position,i))
						
					if r2_word == word:
						augmented = True
						substitution_info[1].append((position,i))
						
					if r3_word == word:
						augmented = True
						substitution_info[2].append((position,i))

			# exclude the sentence pair if cda_mode=='partial' and augmented=False
			if cda_mode=='complete' or augmented:
				occurred_groups = []
				for group,info in substitution_info.items():
					if len(info)>0:
						occurred_groups.append(group)
				substitutions = list(itertools.permutations(range(3),len(occurred_groups))) # generate permuatations of (0,1,2) with len(...)
				substitutions.remove(tuple(occurred_groups)) # remove the identical substitution
				
				outputs.append(sentence)
				(sub1,sub2) = random.sample(substitutions,2)
				for idx,group in enumerate(occurred_groups):
					for pos,i in substitution_info[group]:
						augmented_sentence_1[pos] = bias_attribute_words[i][sub1[idx]]
						augmented_sentence_2[pos] = bias_attribute_words[i][sub2[idx]]
				augmented_sentence_1 = " ".join(augmented_sentence_1)
				augmented_sentence_2 = " ".join(augmented_sentence_2)
				outputs.append(augmented_sentence_1)
				outputs.append(augmented_sentence_2)	
				
				n_groups[len(occurred_groups)] += 1
				if len(occurred_groups)>1:
					print(sub1,sub2)
					print(sentence)
					print(augmented_sentence_1)
					print(augmented_sentence_2)

		print(n_groups)
		# There are potentially no counterfactual examples.
		if not outputs:
			return {"input_ids": [], "attention_mask": []}

		return tokenizer(
			outputs,
			return_special_tokens_mask=True,
			add_special_tokens=False,  # Special tokens are already added.
			truncation=True,
			padding=False, # True
			max_length=max_seq_length,
		)

	if data_args.bias_type is not None:
		if data_args.bias_type not in ["gender", "race", "religion"]:
			raise ValueError("Invalid bias type: {data_args.bias_type")

		logger.info(f"Applying {data_args.bias_type} CDA.")

		# Load the bias attribute words.
		bias_attribute_words = _create_bias_attribute_words(os.path.join('data','bias_attribute_words.json'),
			bias_type=data_args.bias_type)

		if data_args.bias_type == "gender":
			counterfactual_augmentation_func = partial(
				gender_counterfactual_augmentation,
				bias_attribute_words=bias_attribute_words,
				cda_mode=data_args.cda_mode
			)
		else:
			counterfactual_augmentation_func = partial(
				ternary_counterfactual_augmentation,
				bias_attribute_words=bias_attribute_words,
				cda_mode=data_args.cda_mode
			)

		tokenized_datasets = tokenized_datasets.map(
			counterfactual_augmentation_func,
			batched=True,
			num_proc=data_args.preprocessing_num_workers,
			load_from_cache_file= False, # not data_args.overwrite_cache,
			desc=f"Applying counterfactual augmentation",
		)

	return tokenized_datasets

	