from typing import Optional,Union,Tuple

import torch
from torch.nn import CrossEntropyLoss 

from transformers.models.bert.modeling_bert import (
	# BertModelWithHeadsAdaptersMixin,
	ModelWithHeadsAdaptersMixin,
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


@add_start_docstrings("""Bert Model with a `language modeling` head on top.""", BERT_START_DOCSTRING)
class BertForMultipleChoiceMaskedLM(ModelWithHeadsAdaptersMixin, BertPreTrainedModel):

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
		self.cls = BertOnlyMLMHead(config)

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
		prediction_scores = self.cls(
			sequence_output,
			inv_lang_adapter=self.bert.get_invertible_adapter(),
		) # (bsz*n_choices,seq_len,vocab_size)

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




