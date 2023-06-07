from typing import Optional,Union,Tuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss 

from transformers.models.gpt2.modeling_gpt2 import (
	# GPT2ModelWithHeadsAdaptersMixin,
	ModelWithHeadsAdaptersMixin,
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

class GPT2ForMultipleChoiceLMHeadModel(ModelWithHeadsAdaptersMixin, GPT2PreTrainedModel):
	_keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

	def __init__(self, config):
		super().__init__(config)

		self.num_labels = config.num_labels
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
		input_ids: Optional[torch.LongTensor] = None, # (bsz,n_choices,seq_len)
		past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None, 
		attention_mask: Optional[torch.FloatTensor] = None, # (bsz,n_choices,seq_len)
		token_type_ids: Optional[torch.LongTensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		head_mask: Optional[torch.FloatTensor] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		encoder_hidden_states: Optional[torch.Tensor] = None,
		encoder_attention_mask: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.Tensor] = None, # (bsz,n_choices,seq_len)
		answers: Optional[torch.Tensor] = None, # (bsz)
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
		) # last_hidden_state,past_key_values,hidden_states,attentions,cross_attentions 
		
		### process outputs like language modeling ###
		hidden_states = transformer_outputs[0] # (bsz*n_choices,seq_len,hidden_size)

		# Set device for model parallelism
		if self.model_parallel:
			torch.cuda.set_device(self.transformer.first_device)
			hidden_states = hidden_states.to(self.lm_head.weight.device)

		lm_logits = self.lm_head(hidden_states) # (bsz*n_choices,seq_len,vocab_size)

		cls_loss = None
		if labels is not None:
			labels = labels.view(-1, labels.size(-1)) # (bsz*n_choices,seq_len)
			# Shift so that tokens < n predict n
			shift_logits = lm_logits[..., :-1, :].contiguous() # (bsz*n_choices,seq_len-1,vocab_size)
			shift_labels = labels[..., 1:].contiguous() # (bsz*n_choices,seq_len-1)
	
			loss_fct = CrossEntropyLoss(reduction='none')  # -100 index = padding token
			causal_lm_losses = torch.mean(loss_fct(shift_logits.permute([0,2,1]).contiguous(),shift_labels),dim=1,keepdim=False)
			# CE((bsz*n_choices,vocab_size,seq_len-1),(bsz*n_choices,seq_len-1))->(bsz*n_choices,seq_len-1)-mean->(bsz*n_choices)
			# CE((N,C,d1,d2,...,dK),(N,d1,...,dK))->(N,d1,...,dK)
			predicted_nlls = causal_lm_losses.view(-1,num_choices) # (bsz,n_choices)

			# transform the answer indices to answer mask
			answer_mask = (-1)*torch.ones(predicted_nlls.shape,device=predicted_nlls.device).scatter(dim=1,index=answers.unsqueeze(1),value=-1)			
			cls_loss = torch.mean(predicted_nlls*answer_mask) # (bsz,n_choices)->(1)

		if not return_dict:
			output = (predicted_nlls,) + transformer_outputs[1:]
			return ((cls_loss,) + output) if loss is not None else output

		return CausalLMOutputWithCrossAttentions(
			loss=cls_loss, # (1)
			logits=predicted_nlls, # (bsz,n_choices)
			past_key_values=transformer_outputs.past_key_values, # (n_layers*(2*(bsz*n_choices,n_heads,seq_len1,head_size)))
			hidden_states=transformer_outputs.hidden_states, # ((n_layers+1)*(bsz*n_choices,seq_len,hidden_size))
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