# Alpaca-LoRA as a service

Demonstrate Alpaca-LoRA as a Chatbot service with [Alpaca-LoRA](https://github.com/tloen/alpaca-lora) and [Gradio](https://gradio.app/). Main features include:
- it enables batch inference by aggregating requests until the previous requests are finished (fixed at 4)

- it achieves context aware by keeping chatting history with the following string:

```python
f"""
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Input: {input} # Surrounding information to AI
### Instruction: {prompt1} # First instruction/prompt given by user
### Response {response1} # First response on the first prompt by AI
### Instruction: {prompt2} # Second instruction/prompt given by user
### Response: {response2} # Second response on the first prompt by AI
....
"""
```

- it provides an additional script to run various configurations to see how it affects the generation quality and speed

- it currently supports the following Alpaca-LoRA:
  - [tloen/alpaca-lora-7b](https://huggingface.co/tloen/alpaca-lora-7b): the original 7B Alpaca-LoRA checkpoint by tloen
  - [chansung/alpaca-lora-13b](https://huggingface.co/chansung/alpaca-lora-13b): the 13B Alpaca-LoRA checkpoint by myself(chansung) with the same script to tune the original 7B model
  - [chansung/koalpaca-lora-13b](https://huggingface.co/chansung/koalpaca-lora-13b): the 13B Alpaca-LoRA checkpoint by myself(chansung) with the Korean dataset created by [KoAlpaca project](https://github.com/Beomi/KoAlpaca) by Beomi (**TBD**)
  - [chansung/alpaca-lora-30b](https://huggingface.co/chansung/alpaca-lora-30b): the 30B Alpaca-LoRA checkpoint by myself(chansung) with the same script to tune the original 7B model (**TBD**)

## Instructions

0. Prerequisites

Note that the code only works `Python >= 3.9`

```shell
$ conda create -n alpaca-serve python=3.9
$ conda activate alpaca-serve
```

1. Install dependencies
```shell
$ pip install -r requirements.txt
```

2. Run Gradio application
```shell
$ BASE_URL=decapoda-research/llama-7b-hf
$ FINETUNED_CKPT_URL=tloen/alpaca-lora-7b
$
$ python app.py --base_url $BASE_URL --ft_ckpt_url $FINETUNED_CKPT_URL --port 6006
```

## Screenshots

<p align="center">
  <img height="900px" src="https://i.ibb.co/VQqfFv4/2023-03-17-2-19-32.png" />
</p>
