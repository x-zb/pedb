# An Empirical Analysis of Parameter-Efficient Methods for Debiasing Pre-Trained Language Models

## Requirements

## External Datasets

A list of external datasets required by this repository:

Dataset | Download Link | Notes | Download Directory
--------|---------------|-------|-------------------
Wikipedia-2.5 | [Download](https://drive.google.com/file/d/1JSlm8MYDbNjpMPnKbb91T-xZnlWAZmZl/view?usp=sharing) | English Wikipedia dump <br>used for SentenceDebias. | `data/text`
Wikipedia-10 | [Download](https://drive.google.com/file/d/1boQTn44RnHdxWeUKQAlRgQ7xrlQ_Glwo/view?usp=sharing) | English Wikipedia dump <br>used for CDA. | `data`
LAMA | [Download](https://dl.fbaipublicfiles.com/LAMA/data.zip) | The four LAMA datasets. | `LAMA/data`

## Experiments on Bias Mitigation
The scripts to run the debiasing experiments with different parameter-efficient methods are in `scripts/${bias_type`, where `${bias_type}` $\in$ {`gender`,`race`,`religion`}.

For example, to mitigate gender bias in GPT-2 with adapter tuning, copy the script `scripts/gender/run_gpt2_adapter_rf48.sh` to and run it from the root directory of this project. Please note that all the scripts adopt a default seed of 42, and you can change the `--seed` argument to use other seeds.

The bash commands to evaluate the CrowS-Pairs stereotype score, StereoSet stereotype score, WikiText-2 perplexity and StereoSet LM score are in `scripts/evaluate_${bias_type}.sh`. Run the commands therein from the root directory of this project to get the evaluation results.

A copy of all the evaluation results from five random seeds (0, 10, 42, 123, 12345) are in `permutation_test/data.py`. Run the following command to compute the p-value of the corresponding permutation test:
```bash
python stat.py --key_strings 
```

## Experiments on LAMA

## Experiments on WinoBias

## Acknowledgements
This repository makes use of codes from the following repositories:

* [An empirical survey of the effectiveness of debiasing techniques for pre-trained language models](https://github.com/McGill-NLP/bias-bench)
* [Prefix-tuning: Optimizing continuous prompts for generation](https://github.com/XiangLi1999/PrefixTuning)
* [Sustainable modular debiasing of language models](https://aclanthology.org/attachments/2021.findings-emnlp.411.Software.zip)
* [Language models as knowledge bases?](https://github.com/facebookresearch/LAMA)

We thank the authors of the above repositories, as well as the authors whose codes are cited by the above repositories.

