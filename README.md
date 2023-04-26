# 💬🚀 LLM as a Chatbot Service

![](https://github.com/deep-diver/LLM-As-Chatbot/blob/main/assets/preview.png?raw=true)

The purpose of this repository is to let people to use lots of open sourced instruction-following fine-tuned LLM models as a Chatbot service. The currently focused models are `LLaMA based Alpaca`, `StableLM based Alpaca`, `LLaMA based Dolly`, and `Flan based Alpaca`. Because different models behave differently, and different models require to form prompts differently, I made a very simple library [`Ping Pong`](https://github.com/deep-diver/PingPong) for model agnostic conversation and context managements.

🔗 **Demo link**: will host demos soon (7B Alpaca, 13B Alpaca, 7B StableLM)

The **easiest way** to run this project is to use Colab. Just open up the [llm_as_chatbot_in_colab](https://github.com/deep-diver/Alpaca-LoRA-Serve/blob/main/notebooks/llm_as_chatbot_in_colab.ipynb) notebook in Colab (there is a button `open in colab`), and run every cell sequentially. With the standard GPU instance(___T4___), you can run 7B and 13B models. With the premium GPU instance(___A100 40GB___), you can even run 30B model! Screenshot👇🏼 Just note that the connection could be somewhat unstable, so I recommend you to use Colab for development purpose.

![](https://i.ibb.co/hZ3771L/Screen-Shot-2023-03-22-at-9-36-15-PM.png)

### Mode

**1. Stream generation mode**: streaming mode handles multiple requests in a interleaving way with threads. For instance, if there are two users (A and B) are connected, A's request is handled, and then B's request is handled, and then A's request is handled again.... This is because of the nature of streaming mode which generates and `yield` tokens in one by one manner. 

**2. Batch generation mode**: deprecated, but this mode will be revived soon.

### Context management

Different model might have different strategies to manage context, so if you want to know the exact strategies applied to each model, take a look at the [`chats`](https://github.com/deep-diver/LLM-As-Chatbot/tree/main/chats) directory. However, here are the basic ideas that I have come up with initially. I have found long prompts will slow down the generation process a lot eventually, so I thought the prompts should be kept as short as possible while as concise as possible at the same time. In the previous version, I have accumulated all the past conversations, and that didn't go well.

- In every turn of the conversation, the past `N` conversations will be kept. Think about the `N` as a hyper-parameter. As an experiment, currently the past 2-3 conversations are only kept for all models.
- In every turn of the conversation, it summarizes or extract information. The summarized information will be given in the every next turn of conversation.

### Currently supported models
  - [tloen/alpaca-lora-7b](https://huggingface.co/tloen/alpaca-lora-7b): the original 7B Alpaca-LoRA checkpoint by tloen (updated by 4/4/2022)
  - [chansung/alpaca-lora-13b](https://huggingface.co/chansung/alpaca-lora-13b): the 13B Alpaca-LoRA checkpoint by myself (chansung, updated by 4/4/2022)
  - [chansung/alpaca-lora-30b](https://huggingface.co/chansung/alpaca-lora-30b): the 30B Alpaca-LoRA checkpoint by myself (chansung, updated by 4/4/2022)
  - [chansung/alpaca-lora-65b](https://huggingface.co/chansung/alpaca-lora-65b): the 65B Alpaca-LoRA checkpoint by myself (chansung)
  - [stabilityai/stablelm-tuned-alpha-7b](https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b): StableLM based fine-tuned model
  - [beomi/KoAlpaca-Polyglot-12.8B](https://huggingface.co/beomi/KoAlpaca-Polyglot-12.8B): [Polyglot](https://github.com/EleutherAI/polyglot) based Alpaca style instruction fine-tuned model
  - [declare-lab/flan-alpaca-xl](https://huggingface.co/declare-lab/flan-alpaca-xl): Flan XL(3B) based Alpaca style instruction fine-tuned model.
  - [declare-lab/flan-alpaca-xxl](https://huggingface.co/declare-lab/flan-alpaca-xxl): Flan XXL(11B) based Alpaca style instruction fine-tuned model.
  - [OpenAssistant/stablelm-7b-sft-v7-epoch-3](https://huggingface.co/OpenAssistant/stablelm-7b-sft-v7-epoch-3): StableLM(7B) based OpenAssistant's oasst1 instruction fine-tuned model.
  
## Instructions

0. Prerequisites

Note that the code only works `Python >= 3.9`

```console
$ conda create -n llm-serve python=3.9
$ conda activate llm-serve
```

1. Install dependencies. Update `gradio` version as needed(`gradio > 𝟹.𝟸𝟻` will display code blocks correctly)
```console
$ cd LLM-As-Chatbot
$ pip install -r requirements.txt
```

2. Run Gradio application (GUI Menu)

  - You can choose either `2(GUI Menu)` or `3(CLI)` to run this application. With the `2(GUI Menu)`, you don't have to worry about setting up the environment variables and options yourself. However, `2(GUI Menu)` is inteded to be used for personla usage while `3(CLI)` is intended to be used for serving purpose. With `2(GUI Menu)`, you will see the landing page like below.

```console
# it is recommended to use latest version of Gradio
$ pip -U install gradio >= 3.27.0
$ python menu_app.py
```

![](https://github.com/deep-diver/LLM-As-Chatbot/blob/main/assets/guimode_preview.gif?raw=true)

3. Run Gradio application (CLI)
```console
### for Alpaca 7B 
$ BASE_URL=decapoda-research/llama-7b-hf
$ FINETUNED_CKPT_URL=tloen/alpaca-lora-7b
$ GEN_CONFIG=configs/gen_config_default.yaml
$ SUMMARIZE_GEN_CONFIG=configs/gen_config_summarization_default.yaml

$ python app.py --base-url $BASE_URL \
  --ft-ckpt-url $FINETUNED_CKPT_URL \
  --gen-config-path $GEN_CONFIG \
  --gen-config-summarization-path $SUMMARIZE_GEN_CONFIG
  
### for StableLM 7B   
$ BASE_URL=stabilityai/stablelm-tuned-alpha-7b
$ GEN_CONFIG=configs/gen_config_stablelm.yaml
$ SUMMARIZE_GEN_CONFIG=configs/gen_config_summarization_stablelm.yaml

$ python app.py --base-url $BASE_URL \
  --ft-ckpt-url $FINETUNED_CKPT_URL \
  --gen-config-path $GEN_CONFIG \
  --gen-config-summarization-path $SUMMARIZE_GEN_CONFIG
```

the following flags are supported

```console
usage: app.py [-h] [--base-url BASE_URL] [--ft-ckpt-url FT_CKPT_URL] [--port PORT] [--share] [--gen-config-path GEN_CONFIG_PATH] [--gen-config-summarization-path GEN_CONFIG_SUMMARIZATION_PATH] [--multi-gpu] [--force-download_ckpt] [--chat-only-mode]

Gradio Application for LLM as a chatbot service

options:
  -h, --help            show this help message and exit
  --base-url BASE_URL   Hugging Face Hub URL
  --ft-ckpt-url FT_CKPT_URL
                        Hugging Face Hub URL
  --port PORT           PORT number where the app is served
  --share               Create and share temporary endpoint (useful in Colab env)
  --gen-config-path GEN_CONFIG_PATH
                        path to GenerationConfig file
  --gen-config-summarization-path GEN_CONFIG_SUMMARIZATION_PATH
                        path to GenerationConfig file used in context summarization
  --multi-gpu           Enable multi gpu mode. This will force not to use Int8 but float16, so you need to check if your system has enough GPU memory
  --force-download_ckpt
                        Force to download ckpt instead of using cached one
  --chat-only-mode      Only show chatting window. Otherwise, other components will be appeared for more sophisticated control
```

## Todos

- [X] Gradio components to control the configurations of the generation
- [X] `Flan based Alpaca` models
- [ ] `LLaMA based Dolly` models
- [ ] Multiple conversation management
- [ ] Implement server only option w/ FastAPI
- [ ] ChatGPT's plugin like features

## Acknowledgements

- I am thankful to [Jarvislabs.ai](https://jarvislabs.ai/) who generously provided free GPU resources to experiment with Alpaca-LoRA deployment and share it to communities to try out.
- I am thankful to [Common Computer](https://comcom.ai/ko/) who generously provided A100(40G) x 8 DGX workstation for fine-tuning the models.
