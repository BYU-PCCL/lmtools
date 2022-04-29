import numpy as np
import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, StoppingCriteriaList

from lmtools.lmsampler_baseclass import LMSamplerBaseClass
from lmtools.KeywordsStoppingCriteria import KeywordsStoppingCriteria


class LM_GPTNEO(LMSamplerBaseClass):
    def __init__(self, model_name):
        super().__init__(model_name)
        """
        Supported models: 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-125M'
        """
        # check if model name is supported
        if not any(
            str in model_name
            for str in [
                "EleutherAI/gpt-neo-2.7B",
                "EleutherAI/gpt-neo-1.3B",
                "EleutherAI/gpt-neo-125M",
            ]
        ):
            raise ValueError(
                "Model name not supported. Supported models: EleutherAI/gpt-neo-2.7B, EleutherAI/gpt-neo-1.3B, EleutherAI/gpt-neo-125M"
            )
        # initialize model with model_name
        print(f"Loading {model_name}...")
        self.model = GPTNeoForCausalLM.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # get the number of attention layers
        if torch.cuda.is_available():
            # get all available GPUs
            self.device = "cuda:0"
            self.model = self.model.to(self.device)
            print(f"Loaded model on 1 GPU.")
        else:
            self.device = "cpu"
            print("Loaded model on cpu.")

    def send_prompt(self, prompt, n_probs):
        # encode prompt and pass to model
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(inputs)

        # get logits for final word (the prediction) from model output
        logits = output.logits[-1][-1].to("cpu")

        # get 'n_probs' predicted tokens associated with the above logits
        tokens = torch.argsort(logits, descending=True)[:n_probs]

        # decode tokens into text
        preds = self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=True)
        # TODO - better way to do this?
        # Sometimes symbols don't come out great in ascii encoding
        preds = [p.encode("ascii", "ignore").decode("ascii") for p in preds]

        # calculate real probabilities associated with each prediction
        logits_probs = torch.nn.functional.softmax(logits, dim=0)
        probs = torch.argsort(logits_probs, descending=True)[:n_probs]

        # create dictionary and map prediction word to log prob
        self.pred_dict = {}
        for i in range(len(preds)):
            self.pred_dict[preds[i]] = np.log(logits_probs[probs[i]].item())

        return self.pred_dict

    def sample_several(self, prompt, temperature=0, n_tokens=10, stop_tokens=[]):
        if len(stop_tokens) > 0:
            # if the elements of stop_tokens are strings, tokenize
            if isinstance(stop_tokens[0], str):
                stop_tokens = [self.tokenizer.encode(w)[0] for w in stop_tokens]
            stop_criteria = KeywordsStoppingCriteria(stop_tokens)
            criteria_list = StoppingCriteriaList([stop_criteria])
        else:
            criteria_list = StoppingCriteriaList()

        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        if temperature > 0:
            tokens = self.model.generate(
                input_ids=inputs,
                max_new_tokens=n_tokens,
                do_sample=True,
                temperature=temperature,
                stopping_criteria=criteria_list,
            ).to("cpu")
        else:
            tokens = self.model.generate(
                input_ids=inputs, max_new_tokens=n_tokens, temperature=temperature,
                stopping_criteria=criteria_list,
            ).to("cpu")
        # if stop_token is at the end of the generated sequence, remove it
        if tokens[0,-1] in stop_tokens:
            tokens = tokens[:,:-1]
        preds = self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=True)
        return preds[0][len(prompt) + 1 :]


if __name__ == "__main__":

    model = LM_GPTNEO("EleutherAI/gpt-neo-125M")
    text = model.sample_several(
        prompt="What is the capital of France?\nThe capital of France is"
    )
    print(text)
