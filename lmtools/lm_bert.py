import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizer

from lmtools.lmsampler_baseclass import LMSamplerBaseClass


class LM_BERT(LMSamplerBaseClass):
    def __init__(self, model_name):
        super().__init__(model_name)
        """
        Supported models: 'bert-base-uncased', 'bert-base-cased'
        """
        # check if model_name is supported
        if not any(str in model_name for str in ["bert-base-uncased", "bert-base-cased"]):
            raise ValueError(
                "Model name not supported. Must be one of: bert-base-uncased, bert-base-cased"
            )
        # initialize model with model_name
        print(f"Loading {model_name}...")
        # TODO - add GPU support
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # get the number of attention layers
        if torch.cuda.is_available():
            self.device = "cuda:0"
            self.model = self.model.to(self.device)
            print(f"Loaded model on 1 GPU.")
        else:
            self.device = "cpu"
            print("Loaded model on cpu.")

    def send_prompt(self, prompt, n_probs):
        """
        For BERT style prompts, you can put the '[MASK]' token in where you would like the model to predict.
        """
        if "[MASK]" not in prompt:
            # add mask to end of prompt and period after mask for accurate predictions
            bert_prompt = prompt + " " + self.tokenizer.mask_token + "."
        else:
            bert_prompt = prompt

        # encode bert_prompt
        input = self.tokenizer.encode_plus(bert_prompt, return_tensors="pt").to(
            self.device
        )

        # store the masked token index
        mask_index = torch.where(input["input_ids"][0] == self.tokenizer.mask_token_id)

        # get the output from the model
        with torch.no_grad():
            output = self.model(**input)

        logits = output.logits[0, mask_index].to("cpu").reshape(-1)

        # get 'n_probs' predicted tokens associated with the above logits
        tokens = torch.argsort(logits, descending=True)[:n_probs]

        # decode tokens into text
        preds = self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=True)
        # for some reason, bert tokenizer adds a bunch of whitespace. Remove
        preds = [p.replace(" ", "") for p in preds]

        # calculate real probabilities associated with each prediction
        logits_probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = torch.argsort(logits_probs, descending=True)[:n_probs]

        # create dictionary and map prediction word to log prob
        self.pred_dict = {}
        for i in range(len(preds)):
            self.pred_dict[preds[i]] = np.log(logits_probs[probs[i]].item())

        return self.pred_dict
