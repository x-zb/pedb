
from typing import Optional,Union,Tuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss 

from transformers.models.gpt2.modeling_gpt2 import (
	add_start_docstrings,
	GPT2_START_DOCSTRING,
	GPT2PreTrainedModel,
	GPT2Model,
	PARALLELIZE_DOCSTRING,
	add_start_docstrings_to_model_forward,
	GPT2_INPUTS_DOCSTRING,
	add_code_sample_docstrings,
	_TOKENIZER_FOR_DOC,
	_CHECKPOINT_FOR_DOC,
	CausalLMOutputWithCrossAttentions,
	_CONFIG_FOR_DOC,
	DEPARALLELIZE_DOCSTRING
	)

from .prefix_encoder import PrefixEncoder

@add_start_docstrings(
	"""
	The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
	embeddings).
	""",
	GPT2_START_DOCSTRING,
)
class GPT2LMHeadModel(GPT2PreTrainedModel):
	_keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

	def __init__(self, config):
		super().__init__(config)
		self.transformer = GPT2Model(config)
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

		# Model parallel
		self.model_parallel = False
		self.device_map = None

		# Initialize weights and apply final processing
		self.post_init()


	@add_start_docstrings(PARALLELIZE_DOCSTRING)
	def parallelize(self, device_map=None):
		self.device_map = (
			get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
			if device_map is None
			else device_map
		)
		assert_device_map(self.device_map, len(self.transformer.h))
		self.transformer.parallelize(self.device_map)
		self.lm_head = self.lm_head.to(self.transformer.first_device)
		self.model_parallel = True

	@add_start_docstrings(DEPARALLELIZE_DOCSTRING)
	def deparallelize(self):
		self.transformer.deparallelize()
		self.transformer = self.transformer.to("cpu")
		self.lm_head = self.lm_head.to("cpu")
		self.model_parallel = False
		torch.cuda.empty_cache()

	def get_output_embeddings(self):
		return self.lm_head

	def set_output_embeddings(self, new_embeddings):
		self.lm_head = new_embeddings

	def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
		token_type_ids = kwargs.get("token_type_ids", None)
		# only last token for inputs_ids if past is defined in kwargs
		if past:
			input_ids = input_ids[:, -1].unsqueeze(-1)
			if token_type_ids is not None:
				token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

		attention_mask = kwargs.get("attention_mask", None)
		position_ids = kwargs.get("position_ids", None)

		if attention_mask is not None and position_ids is None:
			# create position_ids on the fly for batch generation
			position_ids = attention_mask.long().cumsum(-1) - 1
			position_ids.masked_fill_(attention_mask == 0, 1)
			if past:
				position_ids = position_ids[:, -1].unsqueeze(-1)
		else:
			position_ids = None
		return {
			"input_ids": input_ids,
			"past_key_values": past,
			"use_cache": kwargs.get("use_cache"),
			"position_ids": position_ids,
			"attention_mask": attention_mask,
			"token_type_ids": token_type_ids,
		}

	@add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
	@add_code_sample_docstrings(
		processor_class=_TOKENIZER_FOR_DOC,
		checkpoint=_CHECKPOINT_FOR_DOC,
		output_type=CausalLMOutputWithCrossAttentions,
		config_class=_CONFIG_FOR_DOC,
	)
	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
		attention_mask: Optional[torch.FloatTensor] = None,
		token_type_ids: Optional[torch.LongTensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		head_mask: Optional[torch.FloatTensor] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		encoder_hidden_states: Optional[torch.Tensor] = None,
		encoder_attention_mask: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
		r"""
		labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
			Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
			`labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
			are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
		"""
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		transformer_outputs = self.transformer(
			input_ids,
			past_key_values=past_key_values,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=encoder_attention_mask,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)# last_hidden_state,past_key_values,hidden_states,attentions,cross_attentions 
        # (bsz,seq_len,hidden_size), (n_layers*(2*(bsz,n_heads,seq_len1,head_size))), 
        # ((n_layers+1)*(bsz,seq_len,hidden_size)), (n_layers*(bsz,n_heads,seq_len,seq_len1)),(...)

		hidden_states = transformer_outputs[0] # (bsz,seq_len,hidden_size)

		# Set device for model parallelism
		if self.model_parallel:
			torch.cuda.set_device(self.transformer.first_device)
			hidden_states = hidden_states.to(self.lm_head.weight.device)

		lm_logits = self.lm_head(hidden_states) # (bsz,seq_len,vocab_size)

		loss = None
		if labels is not None:
			# Shift so that tokens < n predict n
			shift_logits = lm_logits[..., :-1, :].contiguous() # (bsz,seq_len-1,vocab_size)
			shift_labels = labels[..., 1:].contiguous() # (bsz,seq_len-1)
			# Flatten the tokens
			loss_fct = CrossEntropyLoss() # deduction = 'mean'
			loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
			# (bsz*(seq_len-1),vocab_size), (bsz*(seq_len-1)) -> ()

		if not return_dict:
			output = (lm_logits,) + transformer_outputs[1:]
			return ((loss,) + output) if loss is not None else output

		return CausalLMOutputWithCrossAttentions(
			loss=loss, # (1)
			logits=lm_logits, # (bsz,seq_len,vocab_size)
			past_key_values=transformer_outputs.past_key_values, # (n_layers*(2*(bsz,n_heads,seq_len1,head_size)))
			hidden_states=transformer_outputs.hidden_states, # ((n_layers+1)*(bsz,seq_len,hidden_size))
			attentions=transformer_outputs.attentions,
			cross_attentions=transformer_outputs.cross_attentions,
		)

	@staticmethod
	def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
		"""
		This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
		[`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
		beam_idx at every generation step.
		"""
		return tuple(
			tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
			for layer_past in past
		)



@add_start_docstrings(
	"""
	The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
	embeddings).
	""",
	GPT2_START_DOCSTRING,
)
class GPT2PrefixTuningLMHeadModel(GPT2PreTrainedModel):
	_keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

	def __init__(self, config):
		super().__init__(config)
		self.transformer = GPT2Model(config)
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

		# Model parallel
		self.model_parallel = False
		self.device_map = None

		######################################################
		for param in self.transformer.parameters():
			param.requires_grad = False
		for param in self.lm_head.parameters():
			param.requires_grad = False

		self.pre_seq_len = config.pre_seq_len if config.prefix_tokens is None else len(config.prefix_tokens)
		self.n_layer = config.num_hidden_layers # read from the config file
		self.n_head = config.num_attention_heads # read from the config file
		self.head_size = config.hidden_size//config.num_attention_heads

		self.prefix_tokens = torch.arange(self.pre_seq_len).long() # different from config.prefix_tokens
		self.prefix_encoder = PrefixEncoder(config)
		self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
		
		################################################################

		# Initialize weights and apply final processing
		self.post_init()

		if config.prefix_tokens is not None:
			self.token_init(config)


	@add_start_docstrings(PARALLELIZE_DOCSTRING)
	def parallelize(self, device_map=None):
		self.device_map = (
			get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
			if device_map is None
			else device_map
		)
		assert_device_map(self.device_map, len(self.transformer.h))
		self.transformer.parallelize(self.device_map)
		self.lm_head = self.lm_head.to(self.transformer.first_device)
		self.model_parallel = True

	@add_start_docstrings(DEPARALLELIZE_DOCSTRING)
	def deparallelize(self):
		self.transformer.deparallelize()
		self.transformer = self.transformer.to("cpu")
		self.lm_head = self.lm_head.to("cpu")
		self.model_parallel = False
		torch.cuda.empty_cache()

	def get_output_embeddings(self):
		return self.lm_head

	def set_output_embeddings(self, new_embeddings):
		self.lm_head = new_embeddings

	def token_init(self,config):
		with torch.no_grad():
			prefix_inputs = torch.LongTensor(config.prefix_tokens).unsqueeze(0).to(self.transformer.device) # (1,pre_seq_len)
			init_val = self.transformer(prefix_inputs,return_dict=True,use_cache=True)
			init_val = init_val.past_key_values # (n_layers*(2*(bsz=1,n_head,pre_seq_len,head_size)))
			init_val = torch.cat([torch.cat([past_key_or_value.permute([0,2,1,3]).view(self.pre_seq_len,-1) for past_key_or_value in init_val[i]],dim=-1) for i in range(self.n_layer)],dim=-1) # (pre_seq_len,n_layer*2*hidden_size)		
		if config.prefix_projection:
			raise NotImplementedError("Currently not support token initialization for reparametrized prefix tuning")
		else:
			self.prefix_encoder.embedding.weight.data = init_val # (pre_seq_len,n_layer*2*hidden_size)

	def get_prompt(self,batch_size):
		prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size,-1).to(self.transformer.device) # (bsz,pre_seq_len)
		past_key_values = self.prefix_encoder(prefix_tokens) # (bsz,pre_seq_len,2*n_layers*hidden_size)
		past_key_values = past_key_values.view(batch_size,self.pre_seq_len,self.n_layer*2,self.n_head,self.head_size) # (bsz,pre_seq_len,2*n_layers,n_head,head_size)
		past_key_values = self.dropout(past_key_values)
		past_key_values = past_key_values.permute([2,0,3,1,4]).split(2,dim=0) # split_size_or_sections=2
		# (n_layers*2,bsz,n_heads,pre_seq_len_len,head_size)->(n_layers*(2,bsz,n_heads,pre_seq_len,head_size))
		return past_key_values # (n_layers*(2,bsz,n_heads,pre_seq_len,head_size))

	@add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
	@add_code_sample_docstrings(
		processor_class=_TOKENIZER_FOR_DOC,
		checkpoint=_CHECKPOINT_FOR_DOC,
		output_type=CausalLMOutputWithCrossAttentions,
		config_class=_CONFIG_FOR_DOC,
	)
	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
		attention_mask: Optional[torch.FloatTensor] = None,
		token_type_ids: Optional[torch.LongTensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		head_mask: Optional[torch.FloatTensor] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		encoder_hidden_states: Optional[torch.Tensor] = None,
		encoder_attention_mask: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
		r"""
		labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
			Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
			`labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
			are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
		"""
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		#####################################
		batch_size = input_ids.shape[0]
		past_key_values = self.get_prompt(batch_size=batch_size) # (n_layers*(2,bsz,n_heads,pre_seq_len,head_size))
		prefix_attention_mask = torch.ones(batch_size,self.pre_seq_len).to(self.transformer.device)
		if attention_mask==None: # assume there are no paddings in the input
			attention_mask = torch.ones(input_ids.shape).to(self.transformer.device) # (bsz,seq_len)
		attention_mask = torch.cat([prefix_attention_mask,attention_mask],dim=1) # (bsz,seq_len+pre_seq_len)
		# note that in GPT2Attention._attn(), a causal_mask will first apply to attn_weights,
		# this input `attention_mask` mainly handles padding tokens if there are any,
		# and will be applied after the causal_mask.
		########################################

		transformer_outputs = self.transformer(
			input_ids,
			past_key_values=past_key_values, # modified
			attention_mask=attention_mask, # modified
			token_type_ids=token_type_ids,
			position_ids=None, # position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=encoder_attention_mask,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)# last_hidden_state,past_key_values,hidden_states,attentions,cross_attentions 
        # (bsz,seq_len,hidden_size), (n_layers*(2*(bsz,n_heads,seq_len1,head_size))), 
        # ((n_layers+1)*(bsz,seq_len,hidden_size)), (n_layers*(bsz,n_heads,seq_len,seq_len1)),(...)

		hidden_states = transformer_outputs[0] # (bsz,seq_len,hidden_size)

		# Set device for model parallelism
		if self.model_parallel:
			torch.cuda.set_device(self.transformer.first_device)
			hidden_states = hidden_states.to(self.lm_head.weight.device)

		lm_logits = self.lm_head(hidden_states) # (bsz,seq_len,vocab_size)

		loss = None
		if labels is not None:
			# Shift so that tokens < n predict n
			shift_logits = lm_logits[..., :-1, :].contiguous()
			shift_labels = labels[..., 1:].contiguous()
			# Flatten the tokens
			loss_fct = CrossEntropyLoss() # deduction = 'mean'
			loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
			# (bsz*(seq_len-1),vocab_size), (bsz*(seq_len-1)) -> ()

		if not return_dict:
			output = (lm_logits,) + transformer_outputs[1:]
			return ((loss,) + output) if loss is not None else output

		return CausalLMOutputWithCrossAttentions(
			loss=loss, # (1)
			logits=lm_logits, # (bsz,seq_len,vocab_size)
			past_key_values=transformer_outputs.past_key_values, # (n_layers*(2*(bsz,n_heads,seq_len1,head_size)))
			hidden_states=transformer_outputs.hidden_states, # ((n_layers+1)*(bsz,seq_len,hidden_size))
			attentions=transformer_outputs.attentions,
			cross_attentions=transformer_outputs.cross_attentions,
		)



@add_start_docstrings(
	"""
	The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
	embeddings).
	""",
	GPT2_START_DOCSTRING,
)
class GPT2PromptTuningLMHeadModel(GPT2PreTrainedModel):
	_keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

	def __init__(self, config):
		super().__init__(config)
		self.transformer = GPT2Model(config)
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

		# Model parallel
		self.model_parallel = False
		self.device_map = None

		############################################
		for param in self.transformer.parameters():
			param.requires_grad = False
		for param in self.lm_head.parameters():
			param.requires_grad = False

		self.pre_seq_len = config.pre_seq_len if config.prefix_tokens is None else len(config.prefix_tokens)
		# self.n_layer = config.num_hidden_layers # read from the config file
		# self.n_head = config.num_attention_heads # read from the config file
		# self.head_size = config.hidden_size//config.num_attention_heads

		self.prefix_tokens = torch.arange(self.pre_seq_len).long() # different from config.prefix_tokens, these are new tokens not in the vocab
		self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, config.hidden_size) # excludes token_type and position embeddings
		# self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

		###########################################################

		# Initialize weights and apply final processing
		self.post_init()

		if config.prefix_tokens is not None:
			self.token_init(config.prefix_tokens)


	@add_start_docstrings(PARALLELIZE_DOCSTRING)
	def parallelize(self, device_map=None):
		self.device_map = (
			get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
			if device_map is None
			else device_map
		)
		assert_device_map(self.device_map, len(self.transformer.h))
		self.transformer.parallelize(self.device_map)
		self.lm_head = self.lm_head.to(self.transformer.first_device)
		self.model_parallel = True

	@add_start_docstrings(DEPARALLELIZE_DOCSTRING)
	def deparallelize(self):
		self.transformer.deparallelize()
		self.transformer = self.transformer.to("cpu")
		self.lm_head = self.lm_head.to("cpu")
		self.model_parallel = False
		torch.cuda.empty_cache()

	def get_output_embeddings(self):
		return self.lm_head

	def set_output_embeddings(self, new_embeddings):
		self.lm_head = new_embeddings

	def token_init(self,prefix_tokens):
		with torch.no_grad():
			init_val = self.transformer.wte(torch.LongTensor(prefix_tokens)) # (pre_seq_len,hidden_size), excludes token_type and position embeddings
		self.prefix_encoder.weight.data = init_val

	def get_prompt(self, batch_size):
		prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.transformer.device) # (bsz,prefix_len)
		prompt_embeds = self.prefix_encoder(prefix_tokens) # (bsz,prefix_len,hidden_size), excludes token_type and position enbeddings
		prompt_output = self.transformer(inputs_embeds=prompt_embeds,return_dict=True,use_cache=True) 
		return prompt_output.past_key_values # (n_layers*(2,bsz,n_heads,pre_seq_len,head_size))


	@add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
	@add_code_sample_docstrings(
		processor_class=_TOKENIZER_FOR_DOC,
		checkpoint=_CHECKPOINT_FOR_DOC,
		output_type=CausalLMOutputWithCrossAttentions,
		config_class=_CONFIG_FOR_DOC,
	)
	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
		attention_mask: Optional[torch.FloatTensor] = None,
		token_type_ids: Optional[torch.LongTensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		head_mask: Optional[torch.FloatTensor] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		encoder_hidden_states: Optional[torch.Tensor] = None,
		encoder_attention_mask: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
		r"""
		labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
			Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
			`labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
			are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
		"""
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		############################################ (same as PrefixTuning)
		batch_size = input_ids.shape[0]
		past_key_values = self.get_prompt(batch_size=batch_size) # (n_layers*(2,bsz,n_heads,pre_seq_len,head_size))
		prefix_attention_mask = torch.ones(batch_size,self.pre_seq_len).to(self.transformer.device)
		if attention_mask==None: # assume there are no paddings in the input
			attention_mask = torch.ones(input_ids.shape).to(self.transformer.device) # (bsz,seq_len)
		attention_mask = torch.cat([prefix_attention_mask,attention_mask],dim=1) # (bsz,seq_len+pre_seq_len)
		# note that in GPT2Attention._attn(), a causal_mask will first apply to attn_weights,
		# this input `attention_mask` mainly handles padding tokens if there are any,
		# and will be applied after the causal_mask.
		########################################

		transformer_outputs = self.transformer(
			input_ids,
			past_key_values=past_key_values, # modified
			attention_mask=attention_mask, # modified
			token_type_ids=token_type_ids,
			position_ids=None, # position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=encoder_attention_mask,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)# last_hidden_state,past_key_values,hidden_states,attentions,cross_attentions 
        # (bsz,seq_len,hidden_size), (n_layers*(2*(bsz,n_heads,seq_len1,head_size))), 
        # ((n_layers+1)*(bsz,seq_len,hidden_size)), (n_layers*(bsz,n_heads,seq_len,seq_len1)),(...)	
		
		hidden_states = transformer_outputs[0] # (bsz,seq_len,hidden_size)

		# Set device for model parallelism
		if self.model_parallel:
			torch.cuda.set_device(self.transformer.first_device)
			hidden_states = hidden_states.to(self.lm_head.weight.device)

		lm_logits = self.lm_head(hidden_states) # (bsz,seq_len,vocab_size)

		loss = None
		if labels is not None:
			# Shift so that tokens < n predict n
			shift_logits = lm_logits[..., :-1, :].contiguous() # (bsz,seq_len-1,vocab_size)
			shift_labels = labels[..., 1:].contiguous() # (bsz,seq_len-1)
			# Flatten the tokens
			loss_fct = CrossEntropyLoss() # deduction = 'mean'
			loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
			# (bsz*(seq_len-1),vocab_size), (bsz*(seq_len-1)) -> ()

		if not return_dict:
			output = (lm_logits,) + transformer_outputs[1:]
			return ((loss,) + output) if loss is not None else output

		return CausalLMOutputWithCrossAttentions(
			loss=loss, # (1)
			logits=lm_logits, # (bsz,seq_len,vocab_size)
			past_key_values=transformer_outputs.past_key_values, # (n_layers*(2*(bsz,n_heads,seq_len1,head_size)))
			hidden_states=transformer_outputs.hidden_states, # ((n_layers+1)*(bsz,seq_len,hidden_size))
			attentions=transformer_outputs.attentions,
			cross_attentions=transformer_outputs.cross_attentions,
		)


@add_start_docstrings(
	"""
	The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
	embeddings).
	""",
	GPT2_START_DOCSTRING,
)
class GPT2PromptingLMHeadModel(GPT2PreTrainedModel):
	'''
	Note that there are no trainable parameters in this model. Only use this model for evaluation.
	'''

	_keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

	def __init__(self, config):
		super().__init__(config)
		self.transformer = GPT2Model(config)
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

		# Model parallel
		self.model_parallel = False
		self.device_map = None

		############################################
		for param in self.transformer.parameters():
			param.requires_grad = False
		for param in self.lm_head.parameters():
			param.requires_grad = False

		self.pre_seq_len = len(config.prefix_tokens)
		# self.n_layer = config.num_hidden_layers # read from the config file
		# self.n_head = config.num_attention_heads # read from the config file
		# self.head_size = config.hidden_size//config.num_attention_heads

		self.prefix_tokens = torch.tensor(config.prefix_tokens,dtype=torch.long)
		# self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, config.hidden_size)
		# self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
		
		###########################################################

		# Initialize weights and apply final processing
		self.post_init()


	@add_start_docstrings(PARALLELIZE_DOCSTRING)
	def parallelize(self, device_map=None):
		self.device_map = (
			get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
			if device_map is None
			else device_map
		)
		assert_device_map(self.device_map, len(self.transformer.h))
		self.transformer.parallelize(self.device_map)
		self.lm_head = self.lm_head.to(self.transformer.first_device)
		self.model_parallel = True

	@add_start_docstrings(DEPARALLELIZE_DOCSTRING)
	def deparallelize(self):
		self.transformer.deparallelize()
		self.transformer = self.transformer.to("cpu")
		self.lm_head = self.lm_head.to("cpu")
		self.model_parallel = False
		torch.cuda.empty_cache()

	def get_output_embeddings(self):
		return self.lm_head

	def set_output_embeddings(self, new_embeddings):
		self.lm_head = new_embeddings

	def get_prompt(self, batch_size):
		prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.transformer.device) # (bsz,prefix_len)
		prompt_output = self.transformer(input_ids=prefix_tokens,return_dict=True,use_cache=True) 
		return prompt_output.past_key_values # (n_layers*(2,bsz,n_heads,pre_seq_len,head_size))

	@add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
	@add_code_sample_docstrings(
		processor_class=_TOKENIZER_FOR_DOC,
		checkpoint=_CHECKPOINT_FOR_DOC,
		output_type=CausalLMOutputWithCrossAttentions,
		config_class=_CONFIG_FOR_DOC,
	)
	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
		attention_mask: Optional[torch.FloatTensor] = None,
		token_type_ids: Optional[torch.LongTensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		head_mask: Optional[torch.FloatTensor] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		encoder_hidden_states: Optional[torch.Tensor] = None,
		encoder_attention_mask: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
		r"""
		labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
			Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
			`labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
			are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
		"""
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		############################################ (same as PrefixTuning)
		batch_size = input_ids.shape[0]
		past_key_values = self.get_prompt(batch_size=batch_size) # (n_layers*(2,bsz,n_heads,pre_seq_len,head_size))
		prefix_attention_mask = torch.ones(batch_size,self.pre_seq_len).to(self.transformer.device)
		if attention_mask==None: # assume there are no paddings in the input
			attention_mask = torch.ones(input_ids.shape).to(self.transformer.device) # (bsz,seq_len)
		attention_mask = torch.cat([prefix_attention_mask,attention_mask],dim=1) # (bsz,seq_len+pre_seq_len)
		# note that in GPT2Attention._attn(), a causal_mask will first apply to attn_weights,
		# this input `attention_mask` mainly handles padding tokens if there are any, 
		# and will be applied after the causal_mask.
		########################################

		transformer_outputs = self.transformer(
			input_ids,
			past_key_values=past_key_values, # modified
			attention_mask=attention_mask, # modified
			token_type_ids=token_type_ids,
			position_ids=None, # position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=encoder_attention_mask,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)# last_hidden_state,past_key_values,hidden_states,attentions,cross_attentions 
        # (bsz,seq_len,hidden_size), (n_layers*(2*(bsz,n_heads,seq_len1,head_size))), 
        # ((n_layers+1)*(bsz,seq_len,hidden_size)), (n_layers*(bsz,n_heads,seq_len,seq_len1)),(...)	

		hidden_states = transformer_outputs[0] # (bsz,seq_len,hidden_size)

		# Set device for model parallelism
		if self.model_parallel:
			torch.cuda.set_device(self.transformer.first_device)
			hidden_states = hidden_states.to(self.lm_head.weight.device)

		lm_logits = self.lm_head(hidden_states) # (bsz,seq_len,vocab_size)

		loss = None
		if labels is not None:
			# Shift so that tokens < n predict n
			shift_logits = lm_logits[..., :-1, :].contiguous() # (bsz,seq_len-1,vocab_size)
			shift_labels = labels[..., 1:].contiguous() # (bsz,seq_len-1)
			# Flatten the tokens
			loss_fct = CrossEntropyLoss() # deduction = 'mean'
			loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
			# (bsz*(seq_len-1),vocab_size), (bsz*(seq_len-1)) -> ()

		if not return_dict:
			output = (lm_logits,) + transformer_outputs[1:]
			return ((loss,) + output) if loss is not None else output

		return CausalLMOutputWithCrossAttentions(
			loss=loss, # (1)
			logits=lm_logits, # (bsz,seq_len,vocab_size)
			past_key_values=transformer_outputs.past_key_values, # (n_layers*(2*(bsz,n_heads,seq_len1,head_size)))
			hidden_states=transformer_outputs.hidden_states, # ((n_layers+1)*(bsz,seq_len,hidden_size))
			attentions=transformer_outputs.attentions,
			cross_attentions=transformer_outputs.cross_attentions,
		)