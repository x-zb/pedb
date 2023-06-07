# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import transformers
from transformers import AutoConfig,AutoTokenizer,AutoModelForMaskedLM,GPT2LMHeadModel
if hasattr(transformers,'adapters'):
    from transformers.adapters.configuration import AdapterConfig
from model.utils import get_model
from bias_bench.util import generate_experiment_id, _is_generative, _is_self_debias
from bias_bench.model import models

from .bert_connector import Bert
# from .elmo_connector import Elmo
from .gpt_connector import GPT
# from .transformerxl_connector import TransformerXL
# from .roberta_connector import Roberta


def build_model_by_name(lm, args, model_args, data_args, verbose=True):
    """Load a model by name and args.

    Note, args.lm is not used for model selection. args are only passed to the
    model's initializator.
    """
    MODEL_NAME_TO_CLASS = dict(
        # elmo=Elmo,
        bert=Bert,
        gpt=GPT,
        # transformerxl=TransformerXL,
        # roberta=Roberta
    ) # {'elmo':..,'bert':..,'gpt':..}
    if lm not in MODEL_NAME_TO_CLASS:
        raise ValueError("Unrecognized Language Model: %s." % lm)
    if verbose:
        print("Loading %s model..." % lm)

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
        raise ValueError("You are instantiating a new config instance from scratch. Not supported.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": False, # model_args.use_fast_tokenizer,
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
    if model_args.task_type=="causal_lm":
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = config.eos_token_id

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bias_bench_models = ["SentenceDebiasBertForMaskedLM","INLPBertForMaskedLM","SelfDebiasBertForMaskedLM",
        "SentenceDebiasGPT2LMHeadModel","INLPGPT2LMHeadModel","SelfDebiasGPT2LMHeadModel"]
    if model_args.prompt_model in bias_bench_models:
        debiased_model_to_base_model = {
            "SentenceDebiasBertForMaskedLM":'BertModel',
            "INLPBertForMaskedLM":'BertModel',
            "SelfDebiasBertForMaskedLM":'BertModel',
            "SentenceDebiasGPT2LMHeadModel":'GPT2Model',
            "INLPGPT2LMHeadModel":'GPT2Model',
            "SelfDebiasGPT2LMHeadModel":'GPT2Model'}
        kwargs = {}
        if 'SentenceDebias' in model_args.prompt_model:
            bias_direction = "../results/subspace/subspace_m-{}_c-{}_t-{}.pt".format(
                debiased_model_to_base_model[model_args.prompt_model],model_args.model_name_or_path,data_args.bias_type)
            kwargs["bias_direction"] = torch.load(bias_direction)
        if 'INLP' in model_args.prompt_model:
            projection_matrix = "../results/projection_matrix/projection_m-{}_c-{}_t-{}_s-0.pt".format(
                debiased_model_to_base_model[model_args.prompt_model],model_args.model_name_or_path,data_args.bias_type)
            kwargs["projection_matrix"] = torch.load(projection_matrix)
        model = getattr(models, model_args.prompt_model)(model_args.model_name_or_path, **kwargs)
        if _is_self_debias(model_args.prompt_model):
            model._model.eval()
            model._model.to(device)
        else:
            model.eval()
            model.to(device)
    else:
        if model_args.prefix_tokens is not None:
            model_args.prefix_tokens = tokenizer.encode(model_args.prefix_tokens,add_special_tokens=False)
            print('use real word for initialization, prefix length: {}'.format(len(model_args.prefix_tokens)))
        model = get_model(model_args,config)
        # note that for evaluation, `model_args.model_name_or_path` should be set to the checkpoints saved by debias_xxx.py 
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)
        model.eval()
        
    return MODEL_NAME_TO_CLASS[lm](args,tokenizer,model,_is_self_debias(model_args.prompt_model),'race-color' if data_args.bias_type=='race' else data_args.bias_type)


if hasattr(transformers,'adapters'):
    def build_model_by_name_adapter(lm, args, model_args, adapter_args, verbose=True):
        """Load a model by name and args.

        Note, args.lm is not used for model selection. args are only passed to the
        model's initializator.
        """
        MODEL_NAME_TO_CLASS = dict(
            # elmo=Elmo,
            bert=Bert,
            gpt=GPT,
            # transformerxl=TransformerXL,
            # roberta=Roberta
        ) # {'elmo':..,'bert':..,'gpt':..}
        if lm not in MODEL_NAME_TO_CLASS:
            raise ValueError("Unrecognized Language Model: %s." % lm)
        if verbose:
            print("Loading %s model..." % lm)

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
            raise ValueError("You are instantiating a new config instance from scratch. Not supported.")

        tokenizer_kwargs = {
            "cache_dir": model_args.cache_dir,
            "use_fast": False, # model_args.use_fast_tokenizer,
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
        if model_args.task_type=="causal_lm":
            tokenizer.pad_token = tokenizer.eos_token
            config.pad_token_id = config.eos_token_id

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load model
        model_class = AutoModelForMaskedLM if model_args.task_type=="masked_lm" else GPT2LMHeadModel # AutoModelForCausalLM
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        # note that for evaluation, `model_args.model_name_or_path` should also be the name in model hub 
        model.resize_token_embeddings(len(tokenizer))

        # Setup adapters
        task_name = model_args.task_type # modified
        # check if adapter already exists, otherwise add it
        if task_name not in model.config.adapters:
            # resolve the adapter config
            adapter_config = AdapterConfig.load(
                adapter_args.adapter_config,
                # non_linearity=adapter_args.adapter_non_linearity,
                # reduction_factor=adapter_args.adapter_reduction_factor,
            )
            # load a pre-trained adpater from Hub (or saved path) if specified
            if adapter_args.load_adapter:
                model.load_adapter(
                    adapter_args.load_adapter,
                    config=adapter_config,
                    load_as=task_name,
                    # with_head=False
                )
            else:
                raise ValueError("should specify a saved adapter via `--load_adapter`")
        # Freeze all model weights except of those of this adapter
        # model.train_adapter([task_name])
        # Set the adapters to be used in every forward pass
        model.set_active_adapters(task_name)
        model.to(device)
        model.eval()
            
        return MODEL_NAME_TO_CLASS[lm](args,tokenizer,model)