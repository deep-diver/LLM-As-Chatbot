# 💬🚀 LLM as a Chatbot Service

![](https://i.ibb.co/tMrwNTt/2023-05-24-7-34-27.png)

The purpose of this repository is to let people to use lots of open sourced instruction-following fine-tuned LLM models as a Chatbot service. Because different models behave differently, and different models require differently formmated prompts, I made a very simple library [`Ping Pong`](https://github.com/deep-diver/PingPong) for model agnostic conversation and context managements. Also, I made [`GradioChat`](https://github.com/deep-diver/gradio-chat) UI looking similar to [HuggingChat](https://huggingface.co/chat/) but entirely built in Gradio. Those two projects are fully integrated to power this project. 

### Context management

Different model might have different strategies to manage context, so if you want to know the exact strategies applied to each model, take a look at the [`chats`](https://github.com/deep-diver/LLM-As-Chatbot/tree/main/chats) directory. However, here are the basic ideas that I have come up with initially. I have found long prompts will slow down the generation process a lot eventually, so I thought the prompts should be kept as short as possible while as concise as possible at the same time. In the previous version, I have accumulated all the past conversations, and that didn't go well.

- In every turn of the conversation, the past `N` conversations will be kept. Think about the `N` as a hyper-parameter. As an experiment, currently the past 2-3 conversations are only kept for all models.
- (TBD) In every turn of the conversation, it summarizes or extract information. The summarized information will be given in the every next turn of conversation.

### Currently supported models

<details><summary>Checkout the list of models</summary>

  - [tloen/alpaca-lora-7b](https://huggingface.co/tloen/alpaca-lora-7b): the original 7B Alpaca-LoRA checkpoint by tloen (updated by 4/4/2022)
  - [LLMs/Alpaca-LoRA-7B-elina](https://huggingface.co/LLMs/Alpaca-LoRA-7B-elina): the 7B Alpaca-LoRA checkpoint by Chansung (updated by 5/1/2022)
  - [LLMs/Alpaca-LoRA-13B-elina](https://huggingface.co/LLMs/Alpaca-LoRA-13B-elina): the 13B Alpaca-LoRA checkpoint by Chansung (updated by 5/1/2022)
  - [LLMs/Alpaca-LoRA-30B-elina](https://huggingface.co/LLMs/Alpaca-LoRA-30B-elina): the 30B Alpaca-LoRA checkpoint by Chansung (updated by 5/1/2022)
  - [LLMs/Alpaca-LoRA-65B-elina](https://huggingface.co/LLMs/Alpaca-LoRA-65B-elina): the 65B Alpaca-LoRA checkpoint by Chansung (updated by 5/1/2022)
  - [LLMs/AlpacaGPT4-LoRA-7B-elina](https://huggingface.co/LLMs/AlpacaGPT4-LoRA-7B-elina): the 7B Alpaca-LoRA checkpoint trained on GPT4 generated Alpaca style dataset by Chansung (updated by 5/1/2022)
  - [LLMs/AlpacaGPT4-LoRA-13B-elina](https://huggingface.co/LLMs/AlpacaGPT4-LoRA-13B-elina): the 13B Alpaca-LoRA checkpoint trained on GPT4 generated Alpaca style dataset by Chansung (updated by 5/1/2022)
  - [stabilityai/stablelm-tuned-alpha-7b](https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b): StableLM based fine-tuned model
  - [beomi/KoAlpaca-Polyglot-12.8B](https://huggingface.co/beomi/KoAlpaca-Polyglot-12.8B): [Polyglot](https://github.com/EleutherAI/polyglot) based Alpaca style instruction fine-tuned model
  - [declare-lab/flan-alpaca-xl](https://huggingface.co/declare-lab/flan-alpaca-xl): Flan XL(3B) based Alpaca style instruction fine-tuned model.
  - [declare-lab/flan-alpaca-xxl](https://huggingface.co/declare-lab/flan-alpaca-xxl): Flan XXL(11B) based Alpaca style instruction fine-tuned model.
  - [OpenAssistant/stablelm-7b-sft-v7-epoch-3](https://huggingface.co/OpenAssistant/stablelm-7b-sft-v7-epoch-3): StableLM(7B) based OpenAssistant's oasst1 instruction fine-tuned model.
  - [Writer/camel-5b-hf](https://huggingface.co/Writer/camel-5b-hf): Palmyra-base based instruction fine-tuned model. The foundation model and the data are from its creator, [Writer](https://dev.writer.com).
  - [lmsys/fastchat-t5-3b-v1.0](https://huggingface.co/lmsys/fastchat-t5-3b-v1.0): T5(3B) based Vicuna style instruction fine-tuned model on SharedGPT by [lm-sys](https://github.com/lm-sys/FastChat) 
  - [LLMs/Stable-Vicuna-13B](https://huggingface.co/LLMs/Stable-Vicuna-13B): Stable Vicuna(13B) from Carpel AI and Stability AI. This is not a delta weight, so use it at your own risk. I will make this repo as private soon and add Hugging Face token field.
  - [LLMs/Vicuna-7b-v1.1](https://huggingface.co/LLMs/Vicuna-7b-v1.1): Vicuna(7B) from FastChat. This is not a delta weight, so use it at your own risk. I will make this repo as private soon and add Hugging Face token field.
  - [LLMs/Vicuna-13b-v1.1](https://huggingface.co/LLMs/Vicuna-13b-v1.1): Vicuna(13B) from FastChat. This is not a delta weight, so use it at your own risk. I will make this repo as private soon and add Hugging Face token field.
  - [togethercomputer/RedPajama-INCITE-Chat-7B-v0.1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-7B-v0.1): RedPajama INCITE Chat(7B) from Together.
  - [mosaicml/mpt-7b-chat](https://huggingface.co/mosaicml/mpt-7b-chat): MPT-7B from MOSAIC ML.
  - [teknium/llama-deus-7b-v3-lora](https://huggingface.co/teknium/llama-deus-7b-v3-lora): LLaMA 7B based Alpaca style instruction fine-tuned model. The only difference between Alpaca is that this model is fine-tuned on more data including Alpaca dataset, GPTeacher, General Instruct, Code Instruct, Roleplay Instruct, Roleplay V2 Instruct, GPT4-LLM Uncensored, Unnatural Instructions, WizardLM Uncensored, CamelAI's 20k Biology, 20k Physics, 20k Chemistry, 50k Math GPT4 Datasets, and CodeAlpaca
  - [HuggingFaceH4/starchat-alpha](https://huggingface.co/HuggingFaceH4/starchat-alpha): Starcoder 15.5B based instruction fine-tuned model. This model is particularly good at answering questions about coding. 
  - [LLMs/Vicuna-LoRA-EvolInstruct-7B](https://huggingface.co/LLMs/Vicuna-LoRA-EvolInstruct-7B): LLaMA 7B based Vicuna style instruction fine-tuned model. The dataset to fine-tune this model is from WizardLM's Evol Instruction dataset.
  - [LLMs/Vicuna-LoRA-EvolInstruct-13B](https://huggingface.co/LLMs/Vicuna-LoRA-EvolInstruct-13B): LLaMA 13B based Vicuna style instruction fine-tuned model. The dataset to fine-tune this model is from WizardLM's Evol Instruction dataset.

</details>
  
## Instructions

0. Prerequisites

Note that the code only works `Python >= 3.9` and `gradio >= 3.32.0`

```console
$ conda create -n llm-serve python=3.9
$ conda activate llm-serve
```

1. Install dependencies. `flash-attn` and `triton` are included to support `MPT` models, but it will take a long time to install them. If you don't want to use `MPT`, comment them out.
```console
$ cd LLM-As-Chatbot
$ pip install -r requirements.txt
```

2. Run Gradio application

```console
$ python app.py
```

## How to plugin your own model

You need to follow the following steps to bring your own models in this project.

1. Add your model spec in [`model_cards.json`](https://github.com/deep-diver/LLM-As-Chatbot/blob/main/model_cards.json). If you don't have thumnail image, just leave it as blank string(`""`).
2. Add the button for your model in [`app.py`](https://github.com/deep-diver/LLM-As-Chatbot/blob/2efbb004a1989483cbdbd57a6d2b808f966f516a/app.py#L405). Don't forget to give it a name in the `gr.Button` and `gr.Markdown`. For placeholders, their names are omitted. Assign the `gr.Button` to a variable with the name of your choice.
3. Add the button variable to the [button list](https://github.com/deep-diver/LLM-As-Chatbot/blob/2efbb004a1989483cbdbd57a6d2b808f966f516a/app.py#L559) in the `app.py`
4. Determine the model type in [`global_vars.py`](https://github.com/deep-diver/LLM-As-Chatbot/blob/2efbb004a1989483cbdbd57a6d2b808f966f516a/global_vars.py#L12). If you think your model is similar to one of the existings, just add a filtering rules(`if-else`) and give it the same name. 
5. (Optional) if your model is totally new one, you need to give a new `model_type` in `global_vars.py`, and make changes accordingly in `utils.py`, and `chats/central.py`. 

## Todos

- [X] Gradio components to control the configurations of the generation
- [X] `Flan based Alpaca` models
- [X] Multiple conversation management
- [ ] Implement server only option w/ FastAPI
- [ ] ChatGPT's plugin like features

## Acknowledgements

- I am thankful to [Jarvislabs.ai](https://jarvislabs.ai/) who generously provided free GPU resources to experiment with Alpaca-LoRA deployment and share it to communities to try out.
- I am thankful to [Common Computer](https://comcom.ai/ko/) who generously provided A100(40G) x 8 DGX workstation for fine-tuning the models.
