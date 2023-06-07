from typing import Optional,Union,Tuple

import torch
from torch.nn import CrossEntropyLoss 

from transformers.models.bert.modeling_bert import (
	BertPreTrainedModel,
	BertModel,
	BertOnlyMLMHead,
	MaskedLMOutput,
	add_start_docstrings,
	BERT_START_DOCSTRING,
	add_start_docstrings_to_model_forward,
	BERT_INPUTS_DOCSTRING,
	add_code_sample_docstrings,
	_TOKENIZER_FOR_DOC,
	_CHECKPOINT_FOR_DOC,
	_CONFIG_FOR_DOC
)

from .prefix_encoder import PrefixEncoder


@add_start_docstrings("""Bert Model with a `language modeling` head on top.""", BERT_START_DOCSTRING)
class BertForMultipleChoiceMaskedLM(BertPreTrainedModel):

	_keys_to_ignore_on_load_unexpected = [r"pooler"]
	_keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

	def __init__(self, config):
		super().__init__(config)

		if config.is_decoder:
			logger.warning(
				"If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
				"bi-directional self-attention."
			)

		self.bert = BertModel(config, add_pooling_layer=False)
		self.cls = BertOnlyMLMHead(config) # two-layer MLP with activation function and LayerNorm, hidden_size->hidden_size->vocab_size

		# Initialize weights and apply final processing
		self.post_init()


	def get_output_embeddings(self):
		return self.cls.predictions.decoder

	def set_output_embeddings(self, new_embeddings):
		self.cls.predictions.decoder = new_embeddings

	@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
	@add_code_sample_docstrings(
		processor_class=_TOKENIZER_FOR_DOC,
		checkpoint=_CHECKPOINT_FOR_DOC,
		output_type=MaskedLMOutput,
		config_class=_CONFIG_FOR_DOC,
		expected_output="'paris'",
		expected_loss=0.88,
	)
	def forward(
		self,
		input_ids: Optional[torch.Tensor] = None, # (bsz,n_choices,seq_len)
		attention_mask: Optional[torch.Tensor] = None, # (bsz,n_choices,seq_len)
		token_type_ids: Optional[torch.Tensor] = None, # (bsz,n_choices,seq_len)
		position_ids: Optional[torch.Tensor] = None,
		head_mask: Optional[torch.Tensor] = None,
		inputs_embeds: Optional[torch.Tensor] = None,
		encoder_hidden_states: Optional[torch.Tensor] = None,
		encoder_attention_mask: Optional[torch.Tensor] = None,
		labels: Optional[torch.Tensor] = None, # (bsz,n_choices,seq_len)
		answers: Optional[torch.Tensor] = None, # (bsz)
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
		r"""
		labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
			Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
			config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
			loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
		"""
		### process inputs like multiple choice ###
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict
		num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

		input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None # (bsz*n_choices,seq_len)
		attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None # (bsz*n_choices,seq_len)
		token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None # (bsz*n_choices,seq_len)
		position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None # (bsz*n_choices,seq_len)
		inputs_embeds = (
			inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
			if inputs_embeds is not None
			else None
		) # (bsz*n_choices,seq_len,hidden_size)

		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=encoder_attention_mask,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		) # (last_hidden_states,pooler_output,(hidden_states,attentions,cross_attentions,past_key_values))

		
		### process outputs like language modeling ###
		sequence_output = outputs[0] # (bsz*n_choices,seq_len,hidden_size)
		prediction_scores = self.cls(sequence_output) # (bsz*n_choices,seq_len,vocab_size)

		cls_loss = None
		if labels is not None and answers is not None:
			labels = labels.view(-1, labels.size(-1)) # (bsz*n_choices,seq_len)
			loss_fct = CrossEntropyLoss(reduction='none')  # -100 index = padding token
			masked_lm_losses = torch.mean(loss_fct(prediction_scores.permute([0,2,1]).contiguous(),labels),dim=1,keepdim=False)
			# CE((bsz*n_choices,vocab_size,seq_len),(bsz*n_choices,seq_len))->(bsz*n_choices,seq_len)-mean->(bsz*n_choices)
			# CE((N,C,d1,d2,...,dK),(N,d1,...,dK))->(N,d1,...,dK)
			predicted_nlls = masked_lm_losses.view(-1,num_choices) # (bsz,n_choices)

			# transform the answer indices to answer mask
			answer_mask = (-1)*torch.ones(predicted_nlls.shape,device=predicted_nlls.device).scatter(dim=1,index=answers.unsqueeze(1),value=-1)		
			cls_loss = torch.mean(predicted_nlls*answer_mask) # (bsz,n_choices)->(1)


		if not return_dict:
			output = (predicted_nlls,) + outputs[2:]
			return ((cls_loss,) + output) if cls_loss is not None else output

		return MaskedLMOutput(
			loss=cls_loss, # (1)
			logits=predicted_nlls, # (bsz,n_choices)
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)

	def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
		input_shape = input_ids.shape
		effective_batch_size = input_shape[0]

		#  add a dummy token
		if self.config.pad_token_id is None:
			raise ValueError("The PAD token should be defined for generation")

		attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
		dummy_token = torch.full(
			(effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
		)
		input_ids = torch.cat([input_ids, dummy_token], dim=1)

		return {"input_ids": input_ids, "attention_mask": attention_mask}


class BertPrefixTuningForMultipleChoiceMaskedLM(BertPreTrainedModel):

	_keys_to_ignore_on_load_unexpected = [r"pooler"]
	_keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

	def __init__(self, config):
		super().__init__(config)

		if config.is_decoder:
			logger.warning(
				"If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
				"bi-directional self-attention."
			)

		self.bert = BertModel(config, add_pooling_layer=False)
		self.cls = BertOnlyMLMHead(config) # two-layer MLP with activation function and LayerNorm, hidden_size->hidden_size->vocab_size
		############################################
		for param in self.bert.parameters():
			param.requires_grad = False
		for param in self.cls.parameters():
			param.requires_grad = False

		self.pre_seq_len = config.pre_seq_len if config.prefix_tokens is None else len(config.prefix_tokens)
		self.n_layer = config.num_hidden_layers # read from the config file
		self.n_head = config.num_attention_heads # read from the config file
		self.head_size = config.hidden_size//config.num_attention_heads

		self.prefix_tokens = torch.arange(self.pre_seq_len).long()
		self.prefix_encoder = PrefixEncoder(config)
		self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

		###########################################################

		# Initialize weights and apply final processing
		self.post_init()

		if config.prefix_tokens is not None:
			self.token_init(config)


	def get_output_embeddings(self):
		return self.cls.predictions.decoder

	def set_output_embeddings(self, new_embeddings):
		self.cls.predictions.decoder = new_embeddings

	def token_init_deprecated(self,config):
		self.bert.config.is_decoder = True
		for module in self.bert.modules():
			if hasattr(module,'is_decoder'):
				module.is_decoder = True
		with torch.no_grad():
			prefix_inputs = torch.LongTensor(config.prefix_tokens).unsqueeze(0).to(self.bert.device) # (1,pre_seq_len)
			init_val = self.bert(prefix_inputs,return_dict=True,use_cache=True)
			init_val = init_val.past_key_values # (n_layers*(2*(bsz=1,n_head,pre_seq_len,head_size)))
			init_val = torch.cat([torch.cat([past_key_or_value.permute([0,2,1,3]).view(self.pre_seq_len,-1) for past_key_or_value in init_val[i]],dim=-1) for i in self.n_layer],dim=-1) # (pre_seq_len,n_layer*2*hidden_size)
		
		self.bert.config.is_decoder = False
		for module in self.bert.modules():
			if hasattr(module,'is_decoder'):
				module.is_decoder = False
		
		if config.prefix_projection:
			raise NotImplementedError("Currently not support token initialization for reparametrized prefix tuning")
		else:
			self.prefix_encoder.embedding.weight.data = init_val # (pre_seq_len,n_layer*2*hidden_size)

	def token_init(self,config):
		with torch.no_grad():
			prefix_inputs = torch.LongTensor(config.prefix_tokens).unsqueeze(0).to(self.bert.device) # (1,pre_seq_len)
			all_hidden_states = self.bert(prefix_inputs,output_hidden_states=True,return_dict=True).hidden_states 
			# ((n_layers+1)*(bsz=1,pre_seq_len,hidden_size))
			
			past_key_values = [] # (n_layers*(2*(bsz=1,n_head,pre_seq_len,head_size)))
			for i in range(self.n_layer):
				self_att = self.bert.encoder.layer[i].attention.self
				key_layer = self_att.transpose_for_scores(self_att.key(all_hidden_states[i])) # (bsz=1,n_heads,pre_seq_len,head_size)
				value_layer = self_att.transpose_for_scores(self_att.value(all_hidden_states[i]))
				past_key_values.append((key_layer,value_layer))
			
			init_val = torch.cat([torch.cat([past_key_or_value.permute([0,2,1,3]).view(self.pre_seq_len,-1) for past_key_or_value in past_key_values[i]],dim=-1) for i in range(self.n_layer)],dim=-1) # (pre_seq_len,n_layer*2*hidden_size)
		
		if config.prefix_projection:
			raise NotImplementedError("Currently not support token initialization for reparametrized prefix tuning")
		else:
			self.prefix_encoder.embedding.weight.data = init_val # (pre_seq_len,n_layer*2*hidden_size)

	def get_prompt(self,batch_size):
		prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size,-1).to(self.bert.device) # (bsz,pre_seq_len)
		past_key_values = self.prefix_encoder(prefix_tokens) # (bsz,pre_seq_len,2*n_layers*hidden_size)
		past_key_values = past_key_values.view(batch_size,self.pre_seq_len,self.n_layer*2,self.n_head,self.head_size) # (bsz,pre_seq_len,2*n_layers,n_head,head_size)
		past_key_values = self.dropout(past_key_values)
		past_key_values = past_key_values.permute([2,0,3,1,4]).split(2,dim=0) # split_size_or_sections=2
		# (n_layers*2,bsz,n_heads,pre_seq_len_len,head_size)->(n_layers*(2,bsz,n_heads,pre_seq_len,head_size))
		return past_key_values # (n_layers*(2,bsz,n_heads,pre_seq_len,head_size))


	@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
	@add_code_sample_docstrings(
		processor_class=_TOKENIZER_FOR_DOC,
		checkpoint=_CHECKPOINT_FOR_DOC,
		output_type=MaskedLMOutput,
		config_class=_CONFIG_FOR_DOC,
		expected_output="'paris'",
		expected_loss=0.88,
	)
	def forward(
		self,
		input_ids: Optional[torch.Tensor] = None, # (bsz,n_choices,seq_len)
		attention_mask: Optional[torch.Tensor] = None, # (bsz,n_choices,seq_len)
		token_type_ids: Optional[torch.Tensor] = None, # (bsz,n_choices,seq_len)
		position_ids: Optional[torch.Tensor] = None,
		head_mask: Optional[torch.Tensor] = None,
		inputs_embeds: Optional[torch.Tensor] = None,
		encoder_hidden_states: Optional[torch.Tensor] = None,
		encoder_attention_mask: Optional[torch.Tensor] = None,
		labels: Optional[torch.Tensor] = None, # (bsz,n_choices,seq_len)
		answers: Optional[torch.Tensor] = None, # (bsz)
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
		r"""
		labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
			Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
			config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
			loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
		"""

		### process inputs like multiple choice ###
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict
		num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

		input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None # (bsz*n_choices,seq_len)
		attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None # (bsz*n_choices,seq_len)
		token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None # (bsz*n_choices,seq_len)
		position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None # (bsz*n_choices,seq_len)
		inputs_embeds = (
			inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
			if inputs_embeds is not None
			else None
		) # (bsz*n_choices,seq_len,hidden_size)

		#####################################
		batch_size = input_ids.shape[0] # bsz*n_choices
		past_key_values = self.get_prompt(batch_size=batch_size) # (n_layers*(2,bsz*n_choices,n_heads,pre_seq_len,head_size))
		prefix_attention_mask = torch.ones(batch_size,self.pre_seq_len).to(self.bert.device) # (bsz*n_choices,pre_seq_len)
		if attention_mask==None: # assume there are no paddings in the input
			attention_mask = torch.ones(input_ids.shape).to(self.bert.device) # (bsz*n_choices,seq_len)
		attention_mask = torch.cat([prefix_attention_mask,attention_mask],dim=1) # (bsz*n_choices,seq_len+pre_seq_len)
		########################################

		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask, # modified
			token_type_ids=token_type_ids,
			position_ids=None, # position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=encoder_attention_mask,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			past_key_values=past_key_values # modified
		) # (last_hidden_states,pooler_output,hidden_states,attentions,cross_attentions,past_key_values)

		### process outputs like language modeling ###
		sequence_output = outputs[0] # (bsz*n_choices,seq_len,hidden_size)
		prediction_scores = self.cls(sequence_output) # (bsz*n_choices,seq_len,vocab_size)

		cls_loss = None
		if labels is not None and answers is not None:
			labels = labels.view(-1, labels.size(-1)) # (bsz*n_choices,seq_len)
			loss_fct = CrossEntropyLoss(reduction='none')  # -100 index = padding token
			masked_lm_losses = torch.mean(loss_fct(prediction_scores.permute([0,2,1]).contiguous(),labels),dim=1,keepdim=False)
			# CE((bsz*n_choices,vocab_size,seq_len),(bsz*n_choices,seq_len))->(bsz*n_choices,seq_len)-mean->(bsz*n_choices)
			# CE((N,C,d1,d2,...,dK),(N,d1,...,dK))->(N,d1,...,dK)
			predicted_nlls = masked_lm_losses.view(-1,num_choices) # (bsz,n_choices)

			# transform the answer indices to answer mask
			answer_mask = (-1)*torch.ones(predicted_nlls.shape,device=predicted_nlls.device).scatter(dim=1,index=answers.unsqueeze(1),value=-1)			
			cls_loss = torch.mean(predicted_nlls*answer_mask) # (bsz,n_choices)->(1)


		if not return_dict:
			output = (predicted_nlls,) + outputs[2:]
			return ((cls_loss,) + output) if cls_loss is not None else output

		return MaskedLMOutput(
			loss=cls_loss, # (1)
			logits=predicted_nlls, # (bsz,n_choices)
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)

class BertPromptTuningForMultipleChoiceMaskedLM(BertPreTrainedModel):

	_keys_to_ignore_on_load_unexpected = [r"pooler"]
	_keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

	def __init__(self, config):
		super().__init__(config)

		if config.is_decoder:
			logger.warning(
				"If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
				"bi-directional self-attention."
			)

		self.bert = BertModel(config, add_pooling_layer=False)
		self.cls = BertOnlyMLMHead(config) # two-layer MLP with activation function and LayerNorm, hidden_size->hidden_size->vocab_size
		
		############################################
		for param in self.bert.parameters():
			param.requires_grad = False
		for param in self.cls.parameters():
			param.requires_grad = False

		self.pre_seq_len = config.pre_seq_len if config.prefix_tokens is None else len(config.prefix_tokens)
		# self.n_layer = config.num_hidden_layers # read from the config file
		# self.n_head = config.num_attention_heads # read from the config file
		# self.head_size = config.hidden_size//config.num_attention_heads

		self.prefix_tokens = torch.arange(self.pre_seq_len).long() # not the same as config.prefix_tokens, these are new tokens not in the vocab
		self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, config.hidden_size) # includes token_type and position embeddings
		self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
		
		###########################################################

		# Initialize weights and apply final processing
		self.post_init()
		
		if config.prefix_tokens is not None:
			self.token_init(config.prefix_tokens)


	def get_output_embeddings(self):
		return self.cls.predictions.decoder

	def set_output_embeddings(self, new_embeddings):
		self.cls.predictions.decoder = new_embeddings

	def token_init(self,prefix_tokens):
		with torch.no_grad():
			init_val = self.bert.embeddings(torch.LongTensor(prefix_tokens)) # (pre_seq_len,hidden_size), includes token_type and position embeddings
		self.prefix_encoder.weight.data = init_val

	def get_prompt(self, batch_size):
		prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device) # (bsz,prefix_len)
		prompts = self.prefix_encoder(prefix_tokens) # (bsz,prefix_len,hidden_size), includes token_type and position enbeddings
		return prompts

	@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
	@add_code_sample_docstrings(
		processor_class=_TOKENIZER_FOR_DOC,
		checkpoint=_CHECKPOINT_FOR_DOC,
		output_type=MaskedLMOutput,
		config_class=_CONFIG_FOR_DOC,
		expected_output="'paris'",
		expected_loss=0.88,
	)
	def forward(
		self,
		input_ids: Optional[torch.Tensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		token_type_ids: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.Tensor] = None,
		head_mask: Optional[torch.Tensor] = None,
		inputs_embeds: Optional[torch.Tensor] = None,
		encoder_hidden_states: Optional[torch.Tensor] = None,
		encoder_attention_mask: Optional[torch.Tensor] = None,
		labels: Optional[torch.Tensor] = None, # (bsz,n_choices,seq_len)
		answers: Optional[torch.Tensor] = None, # (bsz)
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
		r"""
		labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
			Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
			config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
			loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
		"""

		### process inputs like multiple choice ###
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict
		num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

		input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None # (bsz*n_choices,seq_len)
		attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None # (bsz*n_choices,seq_len)
		token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None # (bsz*n_choices,seq_len)
		position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None # (bsz*n_choices,seq_len)


		############################################
		batch_size = input_ids.shape[0] # bsz*n_choices
		
		
		raw_embedding = self.bert.embeddings.word_embeddings(input_ids) # (bsz*n_choices,seq_len,hidden_size)

		prompts = self.get_prompt(batch_size=batch_size) # (bsz*n_choices,prefix_len,hidden_size)
		inputs_embeds = torch.cat((prompts, raw_embedding), dim=1) # (bsz*n_choices,prefix_len+seq_len,hidden_size)
		prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device) # (bsz*n_choices,pre_seq_len)
		if attention_mask==None: # assume there are no paddings in the input
			attention_mask = torch.ones(input_ids.shape).to(self.bert.device) # (bsz*n_choices,seq_len)
		attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1) # (bsz*n_choices,prefix_len+seq_len)

		outputs = self.bert(
			# input_ids,
			attention_mask=attention_mask, # modified
			# token_type_ids=token_type_ids,
			# position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds, # modified
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			# past_key_values=past_key_values,
		) # (last_hidden_state,pooler_output,hidden_states,attentions,cross_attentions,past_key_values)

		
		### process outputs like language modeling ###
		sequence_output = outputs[0] # (bsz*n_choices,pre_seq_len+seq_len,hidden_size)
		sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous() # (bsz*n_choices,seq_len,hidden_size)
		prediction_scores = self.cls(sequence_output) # (bsz*n_choices,seq_len,vocab_size)

		cls_loss = None
		if labels is not None and answers is not None:
			labels = labels.view(-1, labels.size(-1)) # (bsz*n_choices,seq_len)
			loss_fct = CrossEntropyLoss(reduction='none')  # -100 index = padding token
			masked_lm_losses = torch.mean(loss_fct(prediction_scores.permute([0,2,1]).contiguous(),labels),dim=1,keepdim=False)
			# CE((bsz*n_choices,vocab_size,seq_len),(bsz*n_choices,seq_len))->(bsz*n_choices,seq_len)-mean->(bsz*n_choices)
			# CE((N,C,d1,d2,...,dK),(N,d1,...,dK))->(N,d1,...,dK)
			predicted_nlls = masked_lm_losses.view(-1,num_choices) # (bsz,n_choices)

			# transform the answer indices to answer mask
			answer_mask = (-1)*torch.ones(predicted_nlls.shape,device=predicted_nlls.device).scatter(dim=1,index=answers.unsqueeze(1),value=-1)			
			cls_loss = torch.mean(predicted_nlls*answer_mask) # (bsz,n_choices)->(1)


		if not return_dict:
			output = (predicted_nlls,) + outputs[2:]
			return ((cls_loss,) + output) if cls_loss is not None else output

		return MaskedLMOutput(
			loss=cls_loss, # (1)
			logits=predicted_nlls, # (bsz,n_choices)
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)


class BertPromptingForMultipleChoiceMaskedLM(BertPreTrainedModel):
	'''
	Note that there are no trainable parameters in this model. Only use this model for evaluation.
	'''

	_keys_to_ignore_on_load_unexpected = [r"pooler"]
	_keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

	def __init__(self, config):
		super().__init__(config)

		if config.is_decoder:
			logger.warning(
				"If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
				"bi-directional self-attention."
			)

		self.bert = BertModel(config, add_pooling_layer=False)
		self.cls = BertOnlyMLMHead(config) # two-layer MLP with activation function and LayerNorm, hidden_size->hidden_size->vocab_size
		
		############################################
		for param in self.bert.parameters():
			param.requires_grad = False
		for param in self.cls.parameters():
			param.requires_grad = False

		self.pre_seq_len = len(config.prefix_tokens)
		# self.n_layer = config.num_hidden_layers # read from the config file
		# self.n_head = config.num_attention_heads # read from the config file
		# self.head_size = config.hidden_size//config.num_attention_heads

		self.prefix_tokens = torch.tensor(config.prefix_tokens,dtype=torch.long)
		# self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, config.hidden_size)
		self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
		
		###########################################################

		# Initialize weights and apply final processing
		self.post_init()


	def get_output_embeddings(self):
		return self.cls.predictions.decoder

	def set_output_embeddings(self, new_embeddings):
		self.cls.predictions.decoder = new_embeddings

	def get_prompt(self, batch_size):
		prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device) # (bsz,prefix_len)
		# prompts = self.prefix_encoder(prefix_tokens) # (bsz,prefix_len,hidden_size)
		prompts = self.bert.embeddings(input_ids=prefix_tokens) # (bsz,prefix_len,hidden_size), includes token_type and position embeddings
		return prompts


	@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
	@add_code_sample_docstrings(
		processor_class=_TOKENIZER_FOR_DOC,
		checkpoint=_CHECKPOINT_FOR_DOC,
		output_type=MaskedLMOutput,
		config_class=_CONFIG_FOR_DOC,
		expected_output="'paris'",
		expected_loss=0.88,
	)
	def forward(
		self,
		input_ids: Optional[torch.Tensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		token_type_ids: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.Tensor] = None,
		head_mask: Optional[torch.Tensor] = None,
		inputs_embeds: Optional[torch.Tensor] = None,
		encoder_hidden_states: Optional[torch.Tensor] = None,
		encoder_attention_mask: Optional[torch.Tensor] = None,
		labels: Optional[torch.Tensor] = None, # (bsz,n_choices,seq_len)
		answers: Optional[torch.Tensor] = None, # (bsz)
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
		r"""
		labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
			Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
			config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
			loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
		"""

		### process inputs like multiple choice ###
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict
		num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

		input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None # (bsz*n_choices,seq_len)
		attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None # (bsz*n_choices,seq_len)
		token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None # (bsz*n_choices,seq_len)
		position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None # (bsz*n_choices,seq_len)

		############################################
		batch_size = input_ids.shape[0] # bsz*n_choices
		raw_embedding = self.bert.embeddings(
			input_ids=input_ids, # (bsz*n_choices,seq_len)
			position_ids=None, # position_ids,
			token_type_ids=token_type_ids, # (bsz*n_choices,seq_len)
			past_key_values_length=self.pre_seq_len # newly added
		) # (bsz*n_choices,seq_len,hidden_size)
		
		prompts = self.get_prompt(batch_size=batch_size) # (bsz*n_choices,pre_seq_len,hidden_size)
		inputs_embeds = torch.cat((prompts, raw_embedding), dim=1) # (bsz*n_choices,pre_seq_len+seq_len,hidden_size)
		prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device) # (bsz*n_choices,pre_seq_len)
		if attention_mask==None: # assume there are no paddings in the input
			attention_mask = torch.ones(input_ids.shape).to(self.bert.device) # (bsz*n_choices,seq_len)
		attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1) # (bsz*n_choices,prefix_len+seq_len)

		outputs = self.bert(
			# input_ids,
			attention_mask=attention_mask, # modified
			# token_type_ids=token_type_ids,
			# position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds, # modified
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			# past_key_values=past_key_values,
		) # (last_hidden_state,pooler_output,hidden_states,attentions,cross_attentions,past_key_values)

		### process outputs like language modeling ###
		sequence_output = outputs[0] # (bsz*n_choices,pre_seq_len+seq_len,hidden_size)
		sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous() # (bsz*n_choices,seq_len,hidden_size)
		prediction_scores = self.cls(sequence_output) # (bsz*n_choices,seq_len,vocab_size)

		cls_loss = None
		if labels is not None and answers is not None:
			labels = labels.view(-1, labels.size(-1)) # (bsz*n_choices,seq_len)
			loss_fct = CrossEntropyLoss(reduction='none')  # -100 index = padding token
			masked_lm_losses = torch.mean(loss_fct(prediction_scores.permute([0,2,1]).contiguous(),labels),dim=1,keepdim=False)
			# CE((bsz*n_choices,vocab_size,seq_len),(bsz*n_choices,seq_len))->(bsz*n_choices,seq_len)-mean->(bsz*n_choices)
			# CE((N,C,d1,d2,...,dK),(N,d1,...,dK))->(N,d1,...,dK)
			predicted_nlls = masked_lm_losses.view(-1,num_choices) # (bsz,n_choices)

			# transform the answer indices to answer mask
			answer_mask = (-1)*torch.ones(predicted_nlls.shape,device=predicted_nlls.device).scatter(dim=1,index=answers.unsqueeze(1),value=-1)		
			cls_loss = torch.mean(predicted_nlls*answer_mask) # (bsz,n_choices)->(1)


		if not return_dict:
			output = (predicted_nlls,) + outputs[2:]
			return ((cls_loss,) + output) if cls_loss is not None else output

		return MaskedLMOutput(
			loss=cls_loss, # (1)
			logits=predicted_nlls, # (bsz,n_choices)
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)


