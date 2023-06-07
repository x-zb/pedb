def _is_generative(model):
    # Checks if we are running an autoregressive model.
    return model in [
        "GPT2LMHeadModel",
        "SentenceDebiasGPT2LMHeadModel",
        "INLPGPT2LMHeadModel",
        "CDAGPT2LMHeadModel",
        "DropoutGPT2LMHeadModel",
        "SelfDebiasGPT2LMHeadModel",
        "GPT2PrefixTuningLMHeadModel",
        "GPT2PromptTuningLMHeadModel",
        "GPT2PromptingLMHeadModel",
        "GPT2MultipleChoiceLMHeadModel",
        "GPT2PrefixTuningMultipleChoiceLMHeadModel",
        "GPT2PromptTuningMultipleChoiceLMHeadModel",
        "GPT2PromptingMultipleChoiceLMHeadModel"
    ]


def _is_self_debias(model):
    # Checks if we are running a Self-Debias model.
    return model in [
        "SelfDebiasGPT2LMHeadModel",
        "SelfDebiasBertForMaskedLM",
        "SelfDebiasAlbertForMaskedLM",
        "SelfDebiasRobertaForMaskedLM",
    ]
