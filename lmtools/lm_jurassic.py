import os
import time

import requests

from lmtools.lmsampler_baseclass import LMSamplerBaseClass

API_KEY = os.environ.get("AI21_API_KEY")


class LM_JURASSIC(LMSamplerBaseClass):
    def __init__(self, model_name="j1-jumbo"):
        """
        Supported models: 'j1-jumbo', 'j1-large'
        """
        super().__init__(model_name)
        if not API_KEY:
            raise Exception('NO API KEY. PLEASE ENTER UNDER "export AI21_API_KEY=????"')
        self.engine = model_name
        # make sure engine is a valid model
        if self.engine not in ["j1-jumbo", "j1-large"]:
            raise ValueError(
                "Invalid model name. Must be one of: 'j1-jumbo', 'j1-large'"
            )
        # make sure API key is set
        self.API_KEY = API_KEY
        # warn only once
        self.warned = False

    def send_prompt(self, prompt, n_probs=100):
        start_time = time.time()
        max_jurassic_n_probs = 64
        if n_probs > max_jurassic_n_probs:
            n_probs = max_jurassic_n_probs
            if not self.warned:
                print(
                    f"n_probs > {max_jurassic_n_probs}. Setting n_probs to {max_jurassic_n_probs}"
                )
                self.warned = True
        response = requests.post(
            f"https://api.ai21.com/studio/v1/{self.engine}/complete",
            headers={"Authorization": "Bearer " + str(self.API_KEY)},
            json={
                "prompt": prompt,
                "numResults": 1,
                "maxTokens": 1,
                "topKReturn": n_probs,
                "temperature": 0.0,
            },
        )
        # parse response to dictionary
        response_dict = response.json()
        if "completions" not in response_dict:
            print(response_dict)
        logprobs = response_dict["completions"][0]["data"]["tokens"][0]["topTokens"]
        # change list of dict {token, logprob} to dictionary of {token: logprob}
        # for each token, replace '▁' with ' '
        logprobs = {
            token["token"].replace("▁", " "): token["logprob"] for token in logprobs
        }
        # change keys to list
        # sort dictionary by values
        sorted_logprobs = dict(
            sorted(logprobs.items(), key=lambda x: x[1], reverse=True)
        )
        # TODO - automate this?
        now = time.time()
        min_time = 1
        if now - start_time < min_time:
            wait_time = min_time - (now - start_time)
            wait_time = max(wait_time, 0)
            time.sleep(wait_time)
        return sorted_logprobs

    def sample_several(self, prompt, temperature=0, n_tokens=10):
        response = requests.post(
            f"https://api.ai21.com/studio/v1/{self.engine}/complete",
            headers={"Authorization": "Bearer " + str(self.API_KEY)},
            json={
                "prompt": prompt,
                "numResults": 1,
                "maxTokens": n_tokens,
                "temperature": temperature,
            },
        )
        response_dict = response.json()
        text = response_dict["completions"][0]["data"]["text"]
        return text


if __name__ == "__main__":
    lm = LM_JURASSIC("j1-jumbo")
    probs = lm.send_prompt("What is the capital of France?\nThe capital of France is")
    probs = lm.send_prompt(prompt)
    print(probs)
