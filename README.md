# lmtools

This is a repository for interacting with LLMs, especially for doing one-token-response tasks. This is built on top of Huggingface, the OpenAI API, and the AI21 API.

## Installation
To install, run the following commands:
```
git clone git@github.com:BYU-PCCL/lmtools.git
cd lmtools
pip3 install .
```

## Command line usage
To run inference on a file, run

`lm-pipeline -d path/to/dataset.pkl -m model_name`

This should automatically do any postprocessing necessary. However, should you wish to rerun the postprocessing,

`lm-postprocess -f path/to/results.pkl`
