# 🦙 🚀 Alpaca-LoRA as a Chatbot Service

<p align="center">
  <img height="500" src="https://github.com/deep-diver/Alpaca-LoRA-Serve/raw/main/assets/preview.gif" />
  <br/>
  <i>auto summarization</i>
</p>

**UPDATE
- [X] Other model supports ([flan series](https://huggingface.co/declare-lab/flan-alpaca-xl))
  - you can pass `declare-lab/flan-alpaca-base`(220M), `declare-lab/flan-alpaca-large`(770M), `declare-lab/flan-alpaca-xl`(3B), or `declare-lab/flan-alpaca-xxl`(11B) to `--base_url` CLI option(in this case, checkpoint option will be ignored).

**TODO (also contribution requests)**:
- [ ] Save/load button to save conversation
- [ ] Other model supports ([baize](https://github.com/project-baize/baize-chatbot), [StackLLaMA](https://huggingface.co/blog/stackllama))
- [ ] Better context management: other than auto-summarization, retrieving relevant information from the past conversation history based on the current conversation window
- [ ] `transformers`' `Streamer`: `transformers` starts supporting [streaming generation](https://huggingface.co/docs/transformers/main/en/generation_strategies#streaming). It will help us to remove the monkey patched `StreamModel`
- [ ] WebGPT like feature: by referencing [webgpt-cli](https://github.com/mukulpatnaik/webgpt-cli)

🔗 **Demo link**: [Batch Mode with 30B](https://notebooksf.jarvislabs.ai/43j3x9FSS8Tg0sqvMlDgKPo9vsoSTTKRsX4RIdC3tNd6qeQ6ktlA0tyWRAR3fe_l) and [Streaming Mode with 30B](https://notebooksf.jarvislabs.ai/BuOu_VbEuUHb09VEVHhfnFq4-PMhBRVCcfHBRCOrq7c4O9GI4dIGoidvNf76UsRL/) (running on a single A6000 and 3xA6000 instances respectively at [jarvislabs.ai](https://jarvislabs.ai/)), and [Hugging Face Space](https://huggingface.co/spaces/chansung/Alpaca-LoRA-Serve) which runs 13B on A10.

The **easiest way** to run this project is to use Colab. Just open up the [alpaca_lora_in_colab](https://github.com/deep-diver/Alpaca-LoRA-Serve/blob/main/notebooks/alpaca_lora_in_colab.ipynb) notebook in Colab (there is a button `open in colab`), and run every cell sequentially. With the standard GPU instance(___T4___), you can run 7B and 13B models. With the premium GPU instance(___A100 40GB___), you can even run 30B model! Screenshot👇🏼 Just note that the connection could be somewhat unstable, so I recommend you to use Colab for development purpose.

![](https://i.ibb.co/hZ3771L/Screen-Shot-2023-03-22-at-9-36-15-PM.png)

This repository demonstrates Alpaca-LoRA as a Chatbot service with [Alpaca-LoRA](https://github.com/tloen/alpaca-lora) and [Gradio](https://gradio.app/). It comes with the following features:

### Mode

**1. Batch Generation Mode**: batch generation mode aggregates requests up to `batch_size`, and pass the prompts in the requests to the model. It waits the current requests are fully handled. For instance, with `batch_size=4`, if a user sends a request, that is under processing. While it is under processing, if other users are connected, up to 4 requests from the users are aggregated and processed as soon as the current one is done.

**2. Streaming Mode**: streaming mode handles multiple requests in a interleaving way with threads. For instance, if there are two users (A and B) are connected, A's request is handled, and then B's request is handled, and then A's request is handled again.... This is because of the nature of streaming mode which generates and `yield` tokens in one by one manner. 

### Context management

- Alpaca-LoRA as a Chatbot Service manages context in two ways. First of all, it remembers(stores) every history of the conversations by default as in the following code snippet. `context_string` is set to empty, but it could be set manually via the `Context` field on top of the screen.
  - language model has a limit on the number of tokens it can consume at a time. In order to address this issue, appliction automatically summarize the past conversation history and put it as context. If there is already context, they will be concatenated. The constraints can be configured in `configs/constraints.yaml`.

```python
f"""{context_string}

### Instruction: {prompt1} # First instruction/prompt given by user

### Response {response1} # First response on the first prompt by AI

### Instruction: {prompt2} # Second instruction/prompt given by user

### Response: {response2} # Second response on the first prompt by AI
....
"""
```

### misc.

- There is a `continue` button in the middle of screen. What it does is to simply send ___"continue."___ prompt to the model. This is useful if you get incomplete previous response from the model. With the ___"continue."___, the model tries to complete the response. Also, since this is a continuation of the response, the ___"continue."___ prompt will be hidden to make chatting history more natural.

### Currently supported LoRA checkpoints
  - [tloen/alpaca-lora-7b](https://huggingface.co/tloen/alpaca-lora-7b): the original 7B Alpaca-LoRA checkpoint by tloen (updated by 4/4/2022)
  - [chansung/alpaca-lora-13b](https://huggingface.co/chansung/alpaca-lora-13b): the 13B Alpaca-LoRA checkpoint by myself(chansung) with the same script to tune the original 7B model (updated by 4/4/2022)
  - [chansung/koalpaca-lora-13b](https://huggingface.co/chansung/koalpaca-lora-13b): the 13B Alpaca-LoRA checkpoint by myself(chansung) with the Korean dataset created by [KoAlpaca project](https://github.com/Beomi/KoAlpaca) by Beomi. It works for English(user) to Korean(AI) conversations
  - [chansung/alpaca-lora-30b](https://huggingface.co/chansung/alpaca-lora-30b): the 30B Alpaca-LoRA checkpoint by myself(chansung) with the same script to tune the original 7B model (updated by 4/4/2022)
  - [chansung/alpaca-lora-65b](https://huggingface.co/chansung/alpaca-lora-65b): the 65B Alpaca-LoRA checkpoint by myself(chansung) with the same script to tune the original 7B model

## Instructions

0. Prerequisites

Note that the code only works `Python >= 3.9`

```console
$ conda create -n alpaca-serve python=3.9
$ conda activate alpaca-serve
```

1. Install dependencies
```console
$ cd Alpaca-LoRA-Serve
$ pip install -r requirements.txt
```

2. Run Gradio application
```console
$ BASE_URL=decapoda-research/llama-7b-hf
$ FINETUNED_CKPT_URL=tloen/alpaca-lora-7b

$ python app.py --base_url $BASE_URL --ft_ckpt_url $FINETUNED_CKPT_URL --port 6006
```

the following flags are supported

```console
usage: app.py [-h] [--base_url BASE_URL] [--ft_ckpt_url FT_CKPT_URL] [--port PORT] [--batch_size BATCH_SIZE] [--api_open] [--share] [--gen_config_path GEN_CONFIG_PATH] [--gen_config_summarization_path GEN_CONFIG_SUMMARIZATION_PATH]
              [--get_constraints_config_path GET_CONSTRAINTS_CONFIG_PATH] [--multi_gpu] [--force_download_ckpt]

Gradio Application for Alpaca-LoRA as a chatbot service

options:
  -h, --help            show this help message and exit
  --base_url BASE_URL   Hugging Face Hub URL
  --ft_ckpt_url FT_CKPT_URL
                        Hugging Face Hub URL
  --port PORT           PORT number where the app is served
  --batch_size BATCH_SIZE
                        Number of requests to handle at the same time
  --api_open            Open as API
  --share               Create and share temporary endpoint (useful in Colab env)
  --gen_config_path GEN_CONFIG_PATH
                        path to GenerationConfig file used in batch mode
  --gen_config_summarization_path GEN_CONFIG_SUMMARIZATION_PATH
                        path to GenerationConfig file used in context summarization
  --get_constraints_config_path GET_CONSTRAINTS_CONFIG_PATH
                        path to ConstraintsConfig file used to constraint user inputs
  --multi_gpu           Enable multi gpu mode. This will force not to use Int8 but float16, so you need to check if your system has enough GPU memory
  --force_download_ckpt
                        Force to download ckpt instead of using cached one
```

## Acknowledgements

- I am thankful to [Jarvislabs.ai](https://jarvislabs.ai/) who generously provided free GPU resources to experiment with Alpaca-LoRA deployment and share it to communities to try out.
- I am thankful to [Common Computer](https://comcom.ai/ko/) who generously provided A100(40G) x 8 DGX workstation for fine-tuning the models.
