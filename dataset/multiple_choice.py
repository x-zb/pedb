import os
import re
import random
from dataclasses import dataclass
from typing import Optional, Union

import torch

from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from datasets import load_dataset,concatenate_datasets

PRO_STAT = {
		'carpenter': 2,'mechanic':4,'construction worker':4, 'laborer':4, 'driver':6,'sheriff':14,'mover':18, 
		'developer':20, 'farmer':22,'guard':22,'chief':27,'janitor':34,'lawyer':35,'cook':38,'physician':38,
		'CEO':39, 'analyst':41,'manager':43, 'supervisor':44, 'salesperson':48, 'editor':52, 'designer':54,
		'accountant':61,'auditor':61, 'writer':63,'baker':65,'clerk':72,'cashier':73, 'counselor':73, 'attendant':76, 
		'teacher':78, 'tailor':80, 'librarian':84, 'assistant':85, 'cleaner':89, 'housekeeper':89,'nurse':90,
		'receptionist':90, 'hairdresser':92, 'secretary':95
	}
PRONOUNS = ['he','him','his','himself','she','her','herself']

def get_winobias(model_args,data_args,training_args,config,tokenizer):
	cache_dir = model_args.cache_dir
	dataset_config_name = data_args.dataset_config_name # in ['type1','type2']
	validation_split_percentage = data_args.validation_split_percentage
	few_shot = data_args.few_shot
	few_shot_seed = data_args.few_shot_seed
	model_type = config.model_type	
	
	data_files = {
		'dev_anti':os.path.join('data','winobias','anti_stereotyped_'+dataset_config_name+'.txt.dev'),
		'dev_pro':os.path.join('data','winobias','pro_stereotyped_'+dataset_config_name+'.txt.dev'),
		'test_anti':os.path.join('data','winobias','anti_stereotyped_'+dataset_config_name+'.txt.test'),
		'test_pro':os.path.join('data','winobias','pro_stereotyped_'+dataset_config_name+'.txt.test')	
	}
	raw_datasets = load_dataset('text',data_files=data_files,cache_dir=cache_dir)

	def preprocess_function(examples): # provide the full dataset as one batch to preprocess_function
		
		examples = [' '.join(text.split()[1:]) for text in examples['text']]
		
		
		def extract_mentions(text:str): # extract (pron,occupation) from the text
			# text = text.lower()
			mentions = re.findall(r'\[(.+?)\]',text.strip()) # `?` for non-greedy search
			# assert len(mentions)==2
			print(text)
			pron = None
			occupation = None
			candidate = None
			for mention in mentions:
				if mention in PRONOUNS:
					if pron==None:
						pron = mention
					else:
						continue
				else:
					assert mention.split()[0] in ['the','The', 'a','an']
					occupation = ' '.join(mention.split()[1:]) # occupation with the first `the` or `The` removed
					# occupation = mention # lower-cased occupation
			for prof in PRO_STAT:
				if prof in text and prof not in occupation:
					candidate = prof
			assert candidate!=None
			return (pron,occupation,candidate)

		targets = []
		answer_spans = []
		answers = []
		for i,text in enumerate(examples):
			# extract candidate answers
			pron,occupation,candidate = extract_mentions(text)
			# remove `[` and `]` from the text
			text = ''.join(re.split(r'\[|\]',text.strip()))
			# calculate answer_spans
			target = [text+' "'+pron[0].upper()+pron[1:]+'" refers to "the '+mention+'".' for mention in ((occupation,candidate) if i%2==0 else (candidate,occupation))]
			prefix = text+' "'+pron[0].upper()+pron[1:]+'" refers to "the' 
			# here we omit the space after `the`, since there will be an additional 220 token 
			# in the tokenized preifx compared to the tokenized target if we use gpt2 tokenizer 
			# (results from bert tokenizer will be the same whether we omit the space or not)
			cls_token = 1 if model_type=='bert' else 0
			sep_token = 1 if model_type=='bert' else 1 # bert: +1 (CLS) -2 (" and .) = -1; gpt2: -1 (".)
			answer_span = [[len(tokenizer.tokenize(prefix))+cls_token,len(tokenizer.tokenize(tgt))-sep_token] for tgt in target]
			answer_spans.append(answer_span) # [[[],[]],[[],[]],...]
			targets.append(target)
			answers.append(0 if i%2==0 else 1)
		targets = sum(targets,[]) # flatten
		tokenized_examples = tokenizer(targets,truncation=True,add_special_tokens=True) # dict
		output = {k:[v[i:i+2] for i in range(0,len(v),2)] for k, v in tokenized_examples.items()} # unflatten

		labels = []
		for i in range(0,len(examples)):
			
			label0 = [-100]*len(output['input_ids'][i][0])
			label0[answer_spans[i][0][0]:answer_spans[i][0][1]] = output['input_ids'][i][0][answer_spans[i][0][0]:answer_spans[i][0][1]]
			label1 = [-100]*len(output['input_ids'][i][1])
			label1[answer_spans[i][1][0]:answer_spans[i][1][1]] = output['input_ids'][i][1][answer_spans[i][1][0]:answer_spans[i][1][1]]
			
			labels.append([label0,label1])
			if model_type=='bert':
				output['input_ids'][i][0][answer_spans[i][0][0]:answer_spans[i][0][1]] = [tokenizer.mask_token_id]*(answer_spans[i][0][1]-answer_spans[i][0][0])
				output['input_ids'][i][1][answer_spans[i][1][0]:answer_spans[i][1][1]] = [tokenizer.mask_token_id]*(answer_spans[i][1][1]-answer_spans[i][1][0])

		output['labels'] = labels
		output['answers'] = answers
		
		return output 
			
	tokenized_datasets = raw_datasets.map(preprocess_function,batched=True,batch_size=-1,load_from_cache_file=False) # provide the full dataset as one batch to preprocess_function
	total_size = len(tokenized_datasets['dev_anti'])
	if few_shot>0:
		assert few_shot%2==0
		few_shot = int(few_shot/2)
		random.seed(few_shot_seed)
		train_range = int(total_size/2)
		train_indices = random.sample(range(train_range),few_shot)
		val_indices = random.sample(range(train_range,total_size),few_shot)
		random.seed(training_args.seed)
	else:
		validation_size = round(total_size*validation_split_percentage/100)
		train_size = total_size-validation_size
		train_indices = range(train_size)
		val_indices = range(train_size,total_size)
	tokenized_datasets['train'] = concatenate_datasets([tokenized_datasets['dev_anti'].select(train_indices),tokenized_datasets['dev_pro'].select(train_indices)])
	tokenized_datasets['validation'] = concatenate_datasets([tokenized_datasets['dev_anti'].select(val_indices),tokenized_datasets['dev_pro'].select(val_indices)])
	print(f"train set size: {len(tokenized_datasets['train'])}, val set size: {len(tokenized_datasets['validation'])}")
	return tokenized_datasets
	# dict keys: {'text','input_ids','attention_mask','labels','answers',('token_type_ids')}
	# paried fields: 'input_ids','attention_mask','labels','token_type_ids'

@dataclass
class DataCollatorForMultipleChoiceAndTokenClassification:
	"""
	Data collator that will dynamically pad the inputs for multiple choice and token classification received.
	"""

	tokenizer: PreTrainedTokenizerBase
	padding: Union[bool, str, PaddingStrategy] = True # dynamic padding to max length of the batch
	max_length: Optional[int] = None
	pad_to_multiple_of: Optional[int] = None
	label_pad_token_id: int = -100 # modified

	def __call__(self, features): 
		# features is a list of dicts, with unused columns ('text') already removed in the Trainer(remove_unused_columns=True)
		answers = [feature.pop('answers') for feature in features] # the 'answers' column is removed and stored in answers so that other fields can be flattened
		batch_size = len(features)
		num_choices = len(features[0]["input_ids"]) # 2
		flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
		
		
		flattened_features = sum(flattened_features, [])

		# `tokenizer.pad` is a method in PreTrainedTokenizerBase
		# It will only pad "input_ids","attention_mask","token_type_ids" and "special_tokens_mask"
		batch = self.tokenizer.pad( 
			flattened_features,
			padding=self.padding, # True, means pad to the longest sequence in the batch
			max_length=self.max_length,
			pad_to_multiple_of=self.pad_to_multiple_of,
			# Conversion to tensors will fail if we have labels as they are not of the same length yet.
			return_tensors=None # "pt",
		) 

		# pad the flattened labels
		labels = [feature['labels'] for feature in flattened_features]
		sequence_length = torch.tensor(batch["input_ids"]).shape[1] # seq_len
		padding_side = self.tokenizer.padding_side
		if padding_side == "right":
			batch["labels"] = [
				list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
			]
		else:
			batch["labels"] = [
				[self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
			]


		# create tensors for "input_ids","token_type_ids","attention_mask","labels"
		batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()} # (bsz*n_choices,seq_len) each

		# un-flatten
		batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()} # (bsz,n_choices,seq_len) each
		
		# add back the answers 
		batch["answers"] = torch.tensor(answers, dtype=torch.int64) # (bsz)
		
		return batch