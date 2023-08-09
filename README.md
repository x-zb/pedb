# An Empirical Analysis of Parameter-Efficient Methods for Debiasing Pre-Trained Language Models

## Requirements

Main requirements are `adapter-transformers==3.0.1` and `datasets==2.3.2`. A list of all the packages in the conda environment is in `environment.yml`.

## External Datasets

A list of external datasets required by this repository:

Dataset | Download Link | Notes | Download Directory
--------|---------------|-------|-------------------
Wikipedia-2.5 | [Download](https://drive.google.com/file/d/1JSlm8MYDbNjpMPnKbb91T-xZnlWAZmZl/view?usp=sharing) | English Wikipedia dump <br>used for SentenceDebias. | `data/text`
Wikipedia-10 | [Download](https://drive.google.com/file/d/1boQTn44RnHdxWeUKQAlRgQ7xrlQ_Glwo/view?usp=sharing) | English Wikipedia dump <br>used for CDA. | `data`
LAMA | [Download](https://dl.fbaipublicfiles.com/LAMA/data.zip) | The four LAMA datasets (Google-RE, T-REx, ConceptNet and SQuAD). | `LAMA/data`

## Experiments on Bias Mitigation
The scripts to run the debiasing experiments are in `scripts/${bias_type`, where `${bias_type}` $\in$ {`gender`,`race`,`religion`}.

For example, to mitigate gender bias in GPT-2 with adapter tuning, copy the script `scripts/gender/run_gpt2_adapter_rf48.sh` to and run it from the root directory of this project. Please note that all the scripts adopt a default seed of 42, and you can change the `--seed` argument to use other seeds.

The bash commands to evaluate the CrowS-Pairs stereotype score, StereoSet stereotype score, WikiText-2 perplexity and StereoSet LM score are in `scripts/evaluate_${bias_type}.sh`. Run the commands therein from the root directory of this project to get the evaluation results.

<!--A copy of all the evaluation results from five random seeds (0, 10, 42, 123, 12345) are stored in the dict `results_data` in `permutation_test/data.py`. Run the following command to compute the p-value of the corresponding permutation test:
```bash
cd permutation_test
python stat.py --key_strings ${KEY_STRINGS}
```
where `${KEY_STRINGS}` incdicates which pair of values you are going to compare. E.g., `gender-bert-ss-adapter-prefix` means the pair of StereoSet stereotype scores of adapter tuning and prefix tuning on gender-debiased BERT. All the `${KEY_STRINGS}`s used in our paper are listed in the dict `predefined_alternatives` in  `permutation_test/data.py`, which also specifies the relationship of the pair of values in the null hypothesis.-->

## Experiments on LAMA

The bash commands to evaluate the gender-debiased models on the four LAMA datasts are in `scripts/evaluate_lama.sh`.

## Experiments on WinoBias

The scripts to train and evaluate the models on the WinoBias dataset are in `scripts/winobias`. For example, to train and evaluate BERT via adapter tuning on the type-1 examples, copy the script `scripts/winobias/wino1_bert_adapter_rf48.sh` to and run it from the root directory of this project.

## Acknowledgements
This repository makes use of codes from the following repositories:

* [An empirical survey of the effectiveness of debiasing techniques for pre-trained language models](https://github.com/McGill-NLP/bias-bench)
* [Prefix-tuning: Optimizing continuous prompts for generation](https://github.com/XiangLi1999/PrefixTuning)
* [Sustainable modular debiasing of language models](https://aclanthology.org/attachments/2021.findings-emnlp.411.Software.zip)
* [Language models as knowledge bases?](https://github.com/facebookresearch/LAMA)

We thank the authors of the above repositories, as well as the authors whose codes are cited by the above repositories.

## Citation
If you find this repository useful, please cite the following paper:
```
@inproceedings{xie-lukasiewicz-2023-empirical,
    title = "An Empirical Analysis of Parameter-Efficient Methods for Debiasing Pre-Trained Language Models",
    author = "Xie, Zhongbin  and
      Lukasiewicz, Thomas",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.876",
    pages = "15730--15745",
}
```



