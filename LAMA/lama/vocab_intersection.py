# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from lama.modules import build_model_by_name
from tqdm import tqdm
import argparse
import spacy
import lama.modules.base_connector as base

from arguments import get_args



CASED_MODELS = [
  # {
  #   # "FAIRSEQ WIKI103"
  #   "lm": "fairseq",
  #   "data": "pre-trained_language_models/fairseq/wiki103_fconv_lm/",
  #   "fairseq_model_name": "wiki103.pt",
  #   "task": "language_modeling",
  #   "cpu": True,
  #   "output_dictionary_size": -1
  # },
  {
    # "TransformerXL"
    "lm": "transformerxl",
    "transformerxl_model_dir": "pre-trained_language_models/transformerxl/transfo-xl-wt103/",
  },
  {
    # "ELMO ORIGINAL"
    "lm": "elmo",
    "elmo_model_dir": "pre-trained_language_models/elmo/original",
    "elmo_model_name": "elmo_2x4096_512_2048cnn_2xhighway",
    "elmo_vocab_name": "vocab-2016-09-10.txt",
    "elmo_warm_up_cycles": 5
  },
  {
    # "ELMO ORIGINAL 5.5B"
    "lm": "elmo",
    "elmo_model_dir": "pre-trained_language_models/elmo/original5.5B/",
    "elmo_model_name": "elmo_2x4096_512_2048cnn_2xhighway_5.5B",
    "elmo_vocab_name": "vocab-enwiki-news-500000.txt",
    "elmo_warm_up_cycles": 5
  },
  {
    # "BERT BASE CASED"
    "lm": "bert",
    "bert_model_name": "bert-base-cased",
    "bert_model_dir": "pre-trained_language_models/bert/cased_L-12_H-768_A-12/",
    "bert_vocab_name": "vocab.txt"
  },
  {
    # "BERT LARGE CASED"
    "lm" : "bert",
    "bert_model_name": "bert-large-cased",
    "bert_model_dir": "pre-trained_language_models/bert/cased_L-24_H-1024_A-16/",
    "bert_vocab_name": "vocab.txt"
  }
]

CASED_COMMON_VOCAB_FILENAME = "pre-trained_language_models/common_vocab_cased.txt"

LOWERCASED_MODELS = [
 {
   # "BERT BASE UNCASED"
   "lm": "bert",
   "bert_model_name": "bert-base-uncased",
   "bert_model_dir": None,
   "bert_vocab_name": "vocab.txt"
 },
 # {
 #   # "BERT LARGE UNCASED"
 #   "lm": "bert",
 #   "bert_model_name": "bert-large-uncased",
 #   "bert_model_dir": None,
 #   "bert_vocab_name": "vocab.txt"
 # },
 {
    "lm": "gpt",
    "label": "gpt2_small",
    "models_names": ["gpt"],
    "gpt2_model_name": "openai-gpt",
    "gpt2_model_dir": None
 }
]

LOWERCASED_COMMON_VOCAB_FILENAME = os.path.join("..","pre-trained_language_models","common_vocab_lowercased.txt")


def __vocab_intersection(models, filename, model_args):

    vocabularies = []

    for arg_dict in models:

        args = argparse.Namespace(**arg_dict)
        print(args)
        model_args.model_name_or_path = 'gpt2' if args.lm=='gpt' else args.bert_model_name
        model_args.task_type = 'causal_lm' if args.lm=='gpt' else 'masked_lm'
        model = build_model_by_name(args.lm, args, model_args)

        vocabularies.append(model.vocab)
        print(type(model.vocab),len(model.vocab))

    if len(vocabularies) > 0:
        common_vocab = set(vocabularies[0])
        for vocab in vocabularies:
            common_vocab = common_vocab.intersection(set(vocab))

        print('init_vocab_size:',len(common_vocab))
        # no special symbols in common_vocab
        for symbol in base.SPECIAL_SYMBOLS:
            if symbol in common_vocab:
                common_vocab.remove(symbol)

        # remove stop words
        from spacy.lang.en.stop_words import STOP_WORDS # These are actually the most common words in any language (like articles, prepositions, pronouns, conjunctions, etc) and does not add much information to the text. 
        for stop_word in STOP_WORDS:
            if stop_word in common_vocab:
                print('STOP_WORD:',stop_word)
                common_vocab.remove(stop_word)

        common_vocab = list(common_vocab)

        # remove punctuation and symbols
        nlp = spacy.load('en')
        manual_punctuation = ['(', ')', '.', ',']
        new_common_vocab = []
        for i in tqdm(range(len(common_vocab))):
            word = common_vocab[i]
            doc = nlp(word)
            token = doc[0]
            if(len(doc) != 1):
                print(word)
                for idx, tok in enumerate(doc):
                    print("{} - {}".format(idx, tok))
            elif word in manual_punctuation:
                pass
            elif token.pos_ == "PUNCT":
                print("PUNCT: {}".format(word))
            elif token.pos_ == "SYM":
                print("SYM: {}".format(word))
            else:
                new_common_vocab.append(word)
            # print("{} - {}".format(word, token.pos_))
        common_vocab = new_common_vocab

        # store common_vocab on file
        with open(filename, 'w',encoding='utf-8') as f:
            for item in sorted(common_vocab):
                f.write("{}\n".format(item))


def main():
    model_args,data_args,training_args = get_args()
    # cased version
    # __vocab_intersection(CASED_MODELS, CASED_COMMON_VOCAB_FILENAME)
    # lowercased version
    __vocab_intersection(LOWERCASED_MODELS, LOWERCASED_COMMON_VOCAB_FILENAME, model_args)


if __name__ == '__main__':
    main()
