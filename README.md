# An Empirical Analysis of Parameter-Efficient Methods for Debiasing Pre-Trained Language Models

## External Datasets

A list of external datasets required by this repository:

Dataset | Download Link | Notes | Download Directory
--------|---------------|-------|-------------------
Wikipedia-2.5 | [Download](https://drive.google.com/file/d/1JSlm8MYDbNjpMPnKbb91T-xZnlWAZmZl/view?usp=sharing) | English Wikipedia dump <br>used for SentenceDebias. | `data/text`
Wikipedia-10 | [Download](https://drive.google.com/file/d/1boQTn44RnHdxWeUKQAlRgQ7xrlQ_Glwo/view?usp=sharing) | English Wikipedia dump <br>used for CDA. | `data`
LAMA | [Download](https://dl.fbaipublicfiles.com/LAMA/data.zip) | The four LAMA datasets. | `LAMA/data`

## Acknowledgements
This repository makes use of code from the following repositories:

* [Towards Debiasing Sentence Representations](https://github.com/pliang279/sent_debias)
* [StereoSet: Measuring Stereotypical Bias in Pre-trained Language Models](https://github.com/moinnadeem/stereoset)
* [CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models](https://github.com/nyu-mll/crows-pairs)
* [On Measuring Social Biases in Sentence Encoders](https://github.com/w4ngatang/sent-bias)
* [Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection](https://github.com/shauli-ravfogel/nullspace_projection)
* [Towards Understanding and Mitigating Social Biases in Language Models](https://github.com/pliang279/lm_bias)


* [An empirical survey of the effectiveness of debiasing techniques for pre-trained language models](https://github.com/McGill-NLP/bias-bench)
* [Prefix-tuning: Optimizing continuous prompts for generation](https://github.com/XiangLi1999/PrefixTuning)
* [Sustainable modular debiasing of language models][https://aclanthology.org/attachments/2021.findings-emnlp.411.Software.zip]
* [Language models as knowledge bases?](https://github.com/facebookresearch/LAMA)

We thank the authors for making their code publicly available.

