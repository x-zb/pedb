import os
from typing import Optional

from dataclasses import dataclass,field
from transformers import (
	MODEL_FOR_MASKED_LM_MAPPING,
	HfArgumentParser,
	TrainingArguments,
	MultiLingAdapterArguments,
	)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
	"""
	Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
	"""
	model_name_or_path: Optional[str] = field(
		default=None,
		metadata={
			"help": "The model checkpoint for weights initialization."
			"Don't set if you want to train a model from scratch."
		},
	)
	model_type: Optional[str] = field(
		default=None,
		metadata={
			"help": "If training from scratch, pass a model type from the list: "
			+ ", ".join(MODEL_TYPES)
		},
	)
	config_overrides: Optional[str] = field(
		default=None,
		metadata={
			"help": "Override some existing default config settings when a model is trained from scratch. Example: "
			"n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
		},
	)
	config_name: Optional[str] = field(
		default=None,
		metadata={
			"help": "Pretrained config name or path if not the same as model_name"
		},
	)
	tokenizer_name: Optional[str] = field(
		default=None,
		metadata={
			"help": "Pretrained tokenizer name or path if not the same as model_name"
		},
	)
	cache_dir: Optional[str] = field(
		default=None,
		metadata={
			"help": "Where do you want to store the pretrained models downloaded from huggingface.co"
		},
	)
	use_fast_tokenizer: bool = field(
		default=True,
		metadata={
			"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
		},
	)
	model_revision: str = field(
		default="main",
		metadata={
			"help": "The specific model version to use (can be a branch name, tag name or commit id)."
		},
	)
	use_auth_token: bool = field(
		default=False,
		metadata={
			"help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
			"with private models)."
		},
	)
	prompt_model: str = field(
		default='none',
		# choices=['prefix_tuning','prompt_tuning','prompting','none'],
		metadata={
			"help": "Will use prompt tuning during training"
		}
	)
	pre_seq_len: int = field(
		default=4,
		metadata={
			"help": "The length of prompt"
		}
	)
	prefix_projection: bool = field(
		default=False,
		metadata={
			"help": "Apply a two-layer MLP head over the prefix embeddings"
		}
	) 
	prefix_hidden_size: int = field(
		default=512,
		metadata={
			"help": "The hidden size of the MLP projection head in Prefix Encoder if prefix projection is used"
		}
	)
	hidden_dropout_prob: float = field(
		default=0.1,
		metadata={
			"help": "The dropout probability used in the models"
		}
	)
	# prefix_init: str = field(
	# 	default=None,
	# 	metadata={"help":'the token sequence used to initialize the prefix; if None, use random initialization'}
	# )
	prefix_tokens: str = field(
		default=None,
		metadata={"help":'the token sequence used for initializing the prefix'}
	)
	task_type: str = field(
		default=None,
		# choices = ["masked_lm","causal_lm","cls","coref"],
		metadata={"help":'the task type'}
	)
	# from_tf: bool = field(
	# 	default=False,
	# 	# choices = ["masked_lm","causal_lm","classification","coref"],
	# 	metadata={"help":'checkpoint type'}
	# )
	def __post_init__(self):
		if self.config_overrides is not None and (
			self.config_name is not None or self.model_name_or_path is not None
		):
			raise ValueError(
				"--config_overrides can't be used in combination with --config_name or --model_name_or_path"
			)

@dataclass
class DataTrainingArguments:
	"""
	Arguments pertaining to what data we are going to input our model for training and eval.
	"""
	dataset_name: Optional[str] = field(
		default=None,
		metadata={"help": "The name of the dataset to use (via the datasets library)."},
	)
	dataset_config_name: Optional[str] = field(
		default=None,
		metadata={
			"help": "The configuration name of the dataset to use (via the datasets library)."
		},
	)
	train_file: Optional[str] = field(
		default=None, metadata={"help": "The input training data file (a text file)."}
	)
	validation_file: Optional[str] = field(
		default=None,
		metadata={
			"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
		},
	)
	overwrite_cache: bool = field(
		default=False,
		metadata={"help": "Overwrite the cached training and evaluation sets"},
	)
	validation_split_percentage: Optional[int] = field(
		default=5,
		metadata={
			"help": "The percentage of the train set used as validation set in case there's no validation split"
		},
	)
	max_seq_length: Optional[int] = field(
		default=None,
		metadata={
			"help": "The maximum total input sequence length after tokenization. Sequences longer "
			"than this will be truncated."
		},
	)
	preprocessing_num_workers: Optional[int] = field(
		default=1,
		metadata={"help": "The number of processes to use for the preprocessing."},
	)
	mlm_probability: float = field(
		default=0.15,
		metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
	)
	line_by_line: bool = field(
		default=False,
		# action='store_true',
		metadata={
			"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."
		},
	)
	pad_to_max_length: bool = field(
		default=False,
		metadata={
			"help": "Whether to pad all samples to `max_seq_length`. "
			"If False, will pad the samples dynamically when batching to the maximum length in the batch."
		},
	)
	max_train_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of training examples to this "
			"value if set."
		},
	)
	max_eval_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
			"value if set."
		},
	)
	bias_type: Optional[str] = field(
		default=None,
		# choices=["gender", "race", "religion"],
		metadata={
			"help": "What type of counterfactual augmentation to apply. Defaults to `None`."
		},
	)
	# work_dir: str = field(
	# 	default=os.path.dirname(os.path.realpath(__file__)),
	# 	metadata={"help": "Directory where all persistent data will be stored."},
	# )
	cda_mode: Optional[str] = field(
		default="partial",
		# choices=["complete", "partial"],
		metadata={
			"help": "What type of counterfactual augmentation to apply. Defaults to `None`."
		},
	)
	down_sample: Optional[float] = field(
		default=-1,
		metadata={
			"help": "A number in [0,1] indicating the percentage of augmented dataset to use."
		},
	)
	few_shot: Optional[int] = field(
		default=-1,
		metadata={
			"help": "The number of (train/validation) shots in few-shot learning."
		},
	)
	few_shot_seed: Optional[int] = field(
		default=42,
		metadata={
			"help": "The random seed for selecting few-shot training and valiadation examples."
		},
	)
	def __post_init__(self):
		if (
			self.dataset_name is None
			and self.train_file is None
			and self.validation_file is None
		):
			pass
			# raise ValueError(
			# 	"Need either a dataset name or a training/validation file."
			# )
		else:
			if self.train_file is not None:
				extension = self.train_file.split(".")[-1]
				assert extension in [
					"csv",
					"json",
					"txt",
				], "`train_file` should be a csv, a json or a txt file."
			if self.validation_file is not None:
				extension = self.validation_file.split(".")[-1]
				assert extension in [
					"csv",
					"json",
					"txt",
				], "`validation_file` should be a csv, a json or a txt file."

model_hub_names = ["bert-base-uncased","bert-large-uncased","gpt2","gpt2-large"]


def get_args():
	parser = HfArgumentParser((ModelArguments,DataTrainingArguments,TrainingArguments,MultiLingAdapterArguments))
	args = parser.parse_args_into_dataclasses()
	return args