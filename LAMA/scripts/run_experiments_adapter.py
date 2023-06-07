# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from batch_eval_KB_completion import main as run_evaluation
from batch_eval_KB_completion import load_file
from lama.modules import build_model_by_name_adapter
import pprint
import statistics
from os import listdir
import os
from os.path import isfile, join
from shutil import copyfile
from collections import defaultdict


from arguments_adapter import get_args

LMs = [
    {
        "lm": "bert",
        "label": "bert_base",
        "models_names": ["bert"],
        "bert_model_name": "bert-base-uncased",
        "bert_model_dir": None # "pre-trained_language_models/bert/cased_L-12_H-768_A-12",
    },
    {
        "lm": "gpt2",
        "label": "gpt2_small",
        "models_names": ["gpt2"],
        "gpt2_model_name": "gpt2",
        "gpt2_model_dir": None # "pre-trained_language_models/bert/cased_L-12_H-768_A-12",
    },
    {
        "lm": "bert",
        "label": "bert_large",
        "models_names": ["bert"],
        "bert_model_name": "bert-large-uncased",
        "bert_model_dir": None # "pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
    },
]


def run_experiments(
    relations,
    data_path_pre,
    data_path_post,
    model_args,
    adapter_args,
    input_param={
        "lm": "bert",
        "label": "bert_base",
        "models_names": ["bert"],
        "bert_model_name": "bert-base-uncased",
        "bert_model_dir": None # "pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
    },
    use_negated_probes=False,
):
    model = None
    pp = pprint.PrettyPrinter(width=41, compact=True)

    all_Precision1 = []
    all_Precision10 = []
    all_MRR = []
    type_Precision1 = defaultdict(list)
    type_count = defaultdict(list)

    results_file = open("last_results.csv", "w+")

    for relation in relations:
        pp.pprint(relation)
        PARAMETERS = {
            "dataset_filename": "{}{}{}".format(
                data_path_pre, relation["relation"], data_path_post
            ),
            "common_vocab_filename": None, # "pre-trained_language_models/common_vocab_lowercased.txt", # modified
            "template": "",
            "bert_vocab_name": "vocab.txt",
            "batch_size": 32,
            "logdir": "output",
            "full_logdir": "output/results/{}/{}".format(
                input_param["label"], relation["relation"]
            ),
            "lowercase": input_param["lm"]=='bert', # False,
            "max_sentence_length": 100,
            "threads": -1,
            "interactive": False,
            "use_negated_probes": use_negated_probes,
        }

        if "template" in relation:
            PARAMETERS["template"] = relation["template"]
            if use_negated_probes:
                PARAMETERS["template_negated"] = relation["template_negated"]

        PARAMETERS.update(input_param)
        print(PARAMETERS)

        args = argparse.Namespace(**PARAMETERS)

        # see if file exists
        try:
            data = load_file(args.dataset_filename) # load json file to list
        except Exception as e:
            print("Relation {} excluded.".format(relation["relation"]))
            print("Exception: {}".format(e))
            continue

        if model is None:
            [model_type_name] = args.models_names
            model = build_model_by_name_adapter(model_type_name, args, model_args, adapter_args)

        Precision1,Precision10,MRR = run_evaluation(args, shuffle_data=False, model=model)
        print("P@1 : {}".format(Precision1), flush=True)
        all_Precision1.append(Precision1)
        all_Precision10.append(Precision10)
        all_MRR.append(MRR)

        results_file.write(
            "{},{}\n".format(relation["relation"], round(Precision1 * 100, 2))
        )
        results_file.flush()

        if "type" in relation:
            type_Precision1[relation["type"]].append(Precision1)
            data = load_file(PARAMETERS["dataset_filename"])
            type_count[relation["type"]].append(len(data))

    mean_p1 = statistics.mean(all_Precision1)
    mean_p10 = statistics.mean(all_Precision10)
    mean_mrr = statistics.mean(all_MRR)
    print("@@@ {} - mean P@1: {} - mean P@10: {} - mean MRR: {}".format(input_param["label"], mean_p1,mean_p10,mean_mrr))
    results_file.close()

    for t, l in type_Precision1.items():

        print(
            "@@@ ",
            input_param["label"],
            t,
            statistics.mean(l),
            sum(type_count[t]),
            len(type_count[t]),
            flush=True,
        )

    return mean_p1, all_Precision1 # return values are not used in main


def get_TREx_parameters(data_path_pre="data/"):
    relations = load_file("{}relations.jsonl".format(data_path_pre))
    data_path_pre += "TREx/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_GoogleRE_parameters():
    relations = [
        {
            "relation": "place_of_birth",
            "template": "[X] was born in [Y] .",
            "template_negated": "[X] was not born in [Y] .",
        },
        {
            "relation": "date_of_birth",
            "template": "[X] (born [Y]).",
            "template_negated": "[X] (not born [Y]).",
        },
        {
            "relation": "place_of_death",
            "template": "[X] died in [Y] .",
            "template_negated": "[X] did not die in [Y] .",
        },
    ]
    data_path_pre = "data/Google_RE/"
    data_path_post = "_test.jsonl"
    return relations, data_path_pre, data_path_post


def get_ConceptNet_parameters(data_path_pre="data/"):
    relations = [{"relation": "test"}]
    data_path_pre += "ConceptNet/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_Squad_parameters(data_path_pre="data/"):
    relations = [{"relation": "test"}]
    data_path_pre += "Squad/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


# def run_all_LMs(parameters):
#     for ip in LMs:
#         print(ip["label"])
#         run_experiments(*parameters, model_args, input_param=ip, use_negated_probes=False)


if __name__ == "__main__":

    model_args,data_args,training_args,adapter_args = get_args()
    if 'bert' in model_args.model_name_or_path:
        ip = {
            "lm": "bert",
            "label": "bert_base-"+model_args.prompt_model,
            "models_names": ["bert"],
            "bert_model_name": "bert-base-uncased",
            "bert_model_dir": None # "pre-trained_language_models/bert/cased_L-12_H-768_A-12",
        }
    elif 'gpt2' in model_args.model_name_or_path:
        ip = {
            "lm": "gpt",
            "label": "gpt2-"+model_args.prompt_model,
            "models_names": ["gpt"],
            "gpt2_model_name": "openai-gpt",
            "gpt2_model_dir": None # "pre-trained_language_models/bert/cased_L-12_H-768_A-12",
        }

    print("1. Google-RE")
    parameters = get_GoogleRE_parameters()
    # run_all_LMs(parameters)
    run_experiments(*parameters, model_args, adapter_args, input_param=ip, use_negated_probes=False)

    print("2. T-REx")
    parameters = get_TREx_parameters()
    # run_all_LMs(parameters)
    run_experiments(*parameters, model_args, adapter_args, input_param=ip, use_negated_probes=False)

    print("3. ConceptNet")
    parameters = get_ConceptNet_parameters()
    # run_all_LMs(parameters)
    run_experiments(*parameters, model_args, adapter_args, input_param=ip, use_negated_probes=False)

    print("4. SQuAD")
    parameters = get_Squad_parameters()
    # run_all_LMs(parameters)
    run_experiments(*parameters, model_args, adapter_args, input_param=ip, use_negated_probes=False)

