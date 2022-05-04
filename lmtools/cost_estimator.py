from transformers import GPT2Tokenizer
import sys

# need to have pytorch and huggingface transformers installed to run this.
# can just run pip install transformers[torch] if you need both, or
# just pip install transformers if you have torch
def n_tokens(prompt):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    num_tokens = len(tokenizer(prompt)["input_ids"])
    return num_tokens


def cost_approximation(df, engine):
    n_tokens_total = df["prompt"].apply(n_tokens).sum()
    print(f"Total number of tokens: {n_tokens_total}")

    possible_engines = ["davinci", "curie", "babbage", "ada"]
    assert engine in possible_engines, f"{engine} is not a valud engine"

    if engine == "davinci":
        cost = (n_tokens_total / 1000) * 0.0600
    elif engine == "curie":
        cost = (n_tokens_total / 1000) * 0.0060
    elif engine == "babbage":
        cost = (n_tokens_total / 1000) * 0.0012
    else:
        cost = (n_tokens_total / 1000) * 0.0008

    print(f"Cost for engine {engine}: {cost}")
    return cost


if __name__ == "__main__":
    # input prompt followed by engine when running the file
    # example: python3 cost_approximator.py "Hello World" curie
    # should print 1.2e-05
    pass
