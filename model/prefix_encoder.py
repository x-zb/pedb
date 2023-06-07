import torch

class PrefixEncoder(torch.nn.Module):
	
	def __init__(self,config):
		super().__init__()
		self.prefix_projection = config.prefix_projection
		if self.prefix_projection:
			self.embedding = torch.nn.Embedding(config.pre_seq_len,config.hidden_size)
			self.trans = torch.nn.Sequential(
				torch.nn.Linear(config.hidden_size,config.prefix_hidden_size),
				torch.nn.Tanh(),
				torch.nn.Linear(config.prefix_hidden_size,config.num_hidden_layers*2*config.hidden_size)
				)
		else:
			self.embedding = torch.nn.Embedding(config.pre_seq_len,config.num_hidden_layers*2*config.hidden_size)

	
	def forward(self,prefix:torch.Tensor): # (bsz,prefix_len)->(bsz,prefix_len,2*n_layers*hidden_size)
		if self.prefix_projection:
			prefix_tokens = self.embedding(prefix) # (bsz,prefix_len)->(bsz,predix_len,hidden_size)
			past_key_values = self.trans(prefix_tokens) # (bsz,prefix_len,n_layers*2*hidden_size)
		else:
			past_key_values = self.embedding(prefix) # (bsz,prefix_len)->(bsz,prefix_len,n_layers*2*hidden_size)
		return past_key_values
