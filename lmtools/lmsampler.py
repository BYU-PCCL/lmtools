from lmtools.lm_bert import LM_BERT
from lmtools.lm_gpt2 import LM_GPT2
from lmtools.lm_gpt3 import LM_GPT3
from lmtools.lm_gptj import LM_GPTJ
from lmtools.lm_gptneo import LM_GPTNEO
from lmtools.lm_jurassic import LM_JURASSIC
from lmtools.lmsampler_baseclass import LMSamplerBaseClass


class LMSampler(LMSamplerBaseClass):
    """
    Class to wrap all other LMSampler classes. This way, we can instantiate just by passing a model name, and it will initialize the corresponding class.
    """

    def __init__(self, model_name):
        self.model_name = model_name
        super().__init__(model_name)
        """
        Supported models:
            - GPT-3: 'gpt3-ada', 'gpt3-babbage', 'gpt3-curie', 'gpt3-davinci', 'ada', 'babbage', 'curie', 'davinci', 'text-davinci-002'
            - GPT-2: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'distilgpt2'
            - GPT-J: 'EleutherAI/gpt-j-6B'
            - GPT-Neo: 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-125M'
            - BERT: 'bert-base-uncased', 'bert-base-cased'
            - Jurassic: 'j1-jumbo', 'j1-large'
        """
        if model_name in [
            "gpt3-ada",
            "gpt3-babbage",
            "gpt3-curie",
            "gpt3-davinci",
            "ada",
            "babbage",
            "curie",
            "davinci",
            "text-davinci-002",
        ]:
            self.model = LM_GPT3(model_name)
        # if any of 'gpt2', ... 'gpt2-xl' in model_name
        elif any(
            str in model_name
            for str in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "distilgpt2"]
        ):
            self.model = LM_GPT2(model_name)
        elif "EleutherAI/gpt-j-6B" in model_name:
            self.model = LM_GPTJ(model_name)
        elif any(
            str in model_name
            for str in [
                "EleutherAI/gpt-neo-2.7B",
                "EleutherAI/gpt-neo-1.3B",
                "EleutherAI/gpt-neo-125M",
            ]
        ):
            self.model = LM_GPTNEO(model_name)
        # elif model_name in ['bert-base-uncased', 'bert-base-cased']:
        elif any(str in model_name for str in ["bert-base-uncased", "bert-base-cased"]):
            self.model = LM_BERT(model_name)
        elif model_name in ["j1-jumbo", "j1-large"]:
            self.model = LM_JURASSIC(model_name)
        else:
            raise ValueError(f"Model {model_name} not supported.")

    def send_prompt(self, prompt, n_probs=100):
        return self.model.send_prompt(prompt, n_probs)

    def sample_several(self, prompt, temperature=0, n_tokens=10, stop=[]):
        return self.model.sample_several(prompt, temperature, n_tokens, stop)


if __name__ == "__main__":
    model_name = "gpt3-ada"
    sampler = LMSampler(model_name)
    print(
        sampler.sample_several("The capital of France is", temperature=0, n_tokens=10)
    )
