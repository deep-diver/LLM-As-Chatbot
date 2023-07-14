## UPDATE
- **Internet search support**: you can enable **internet search** capability in Gradio application and Discord bot. For gradio, there is a `internet mode` option in the control panel. For discord, you need to specify `--internet` option in your prompt. For both cases, you need a Serper API Key which you can get one from [serper.dev](https://serper.dev/). By signing up, you will get free 2,500 free google searches which is pretty much sufficient for a long-term test.
- **Discord Bot support**: you can serve any model from the model zoo as Discord Bot. Find how to do this in the instruction section below.

# 💬🚀 LLM as a Chatbot Service

The purpose of this repository is to let people to use lots of open sourced instruction-following fine-tuned LLM models as a Chatbot service. Because different models behave differently, and different models require differently formmated prompts, I made a very simple library [`Ping Pong`](https://github.com/deep-diver/PingPong) for model agnostic conversation and context managements. 

Also, I made [`GradioChat`](https://github.com/deep-diver/gradio-chat) UI that has a similar shape to [HuggingChat](https://huggingface.co/chat/) but entirely built in Gradio. Those two projects are fully integrated to power this project. 

## Easiest way to try out ( ✅ Gradio, 🚧 Discord Bot )

This project has become the one of the default framework at [jarvislabs.ai](https://jarvislabs.ai/). Jarvislabs.ai is one of the cloud GPU VM provider with the cheapest GPU prices. Furthermore, all the weights of the supported popular open source LLMs are pre-downloaded. You don't need to waste of your money and time to wait until download hundreds of GBs to try out a collection of LLMs. In less than 10 minutes, you can try out any model. 
- for further instruction how to run Gradio application, please follow the [official documentation](https://jarvislabs.ai/docs/llmchat) on the `llmchat` framework.

## Instructions

### Standalone Gradio app

![](https://i.ibb.co/gW7yKj9/2023-05-26-3-31-06.png)

0. Prerequisites

    Note that the code only works `Python >= 3.9` and `gradio >= 3.32.0`

    ```console
    $ conda create -n llm-serve python=3.9
    $ conda activate llm-serve
    ```

1. Install dependencies. 
    ```console
    $ cd LLM-As-Chatbot
    $ pip install -r requirements.txt
    ```

2. Run Gradio application

    There is no required parameter to run the Gradio application. However, there are some small details worth being noted. When `--local-files-only` is set, application won't try to look up the Hugging Face Hub(remote). Instead, it will only use the files already downloaded and cached.

    Hugging Face libraries stores downloaded contents under `~/.cache` by default, and this application assumes so. However, if you downloaded weights in different location for some reasons, you can set `HF_HOME` environment variable. Find more about the [environment variables here](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)

   In order to leverage **internet search** capability, you need Serper API Key. You can set it manually in the control panel or in CLI. When specifying the Serper API Key in CLI, it will be injected into the corresponding UI control. If you don't have it yet, please get one from [serper.dev](https://serper.dev/). By signing up, you will get free 2,500 free google searches which is pretty much sufficient for a long-term test.

    ```console
    $ python app.py --root-path "" \
                    --local-files-only \
                    --share \
                    --debug \
                    --serper-api-key "YOUR SERPER API KEY"
    ```

### Discord Bot

![](https://i.ibb.co/cJ3yDWh/2023-07-14-1-42-23.png)

0. Prerequisites

    Note that the code only works `Python >= 3.9` 

    ```console
    $ conda create -n llm-serve python=3.9
    $ conda activate llm-serve
    ```

1. Install dependencies. 
    ```console
    $ cd LLM-As-Chatbot
    $ pip install -r requirements.txt
    ```

2. Run Discord Bot application. Choose one of the modes in `--mode-[cpu|mps|8bit|4bit|full-gpu]`. `full-gpu` will be choseon by default(`full` means `half` - consider this as a typo to be fixed later).

    The `--token` is a required parameter, and you can get it from [Discord Developer Portal](https://discord.com/developers/docs/intro). If you have not setup Discord Bot from the Discord Developer Portal yet, please follow [How to Create a Discord Bot Account](https://www.freecodecamp.org/news/create-a-discord-bot-with-python/) section of the tutorial from [freeCodeCamp](https://www.freecodecamp.org/) to get the token.

    The `--model-name` is a required parameter, and you can look around the list of supported models from [`model_cards.json`](https://github.com/deep-diver/LLM-As-Chatbot/blob/main/model_cards.json).

    `--max-workers` is a parameter to determine how many requests to be handled concurrently. This simply defines the value of the `ThreadPoolExecutor`.

    When `--local-files-only` is set, application won't try to look up the Hugging Face Hub(remote). Instead, it will only use the files already downloaded and cached.

   In order to leverage **internet search** capability, you need Serper API Key. If you don't have it yet, please get one from [serper.dev](https://serper.dev/). By signing up, you will get free 2,500 free google searches which is pretty much sufficient for a long-term test. Once you have the Serper API Key, you can specify it in `--serper-api-key` option.
   
    - Hugging Face libraries stores downloaded contents under `~/.cache` by default, and this application assumes so. However, if you downloaded weights in different location for some reasons, you can set `HF_HOME` environment variable. Find more about the [environment variables here](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)    

    ```console
    $ python discord_app.py --token "DISCORD BOT TOKEN" \
                            --model-name "alpaca-lora-7b" \
                            --max-workers 1 \
                            --mode-[cpu|mps|8bit|4bit|full-gpu] \
                            --local_files_only \
                            --serper-api-key "YOUR SERPER API KEY"
    ```

4. Supported Discord Bot commands

    There is no slash commands. The only way to interact with the deployed discord bot is to mention the bot. However, you can pass some special strings while mentioning the bot.

    - **`@bot_name help`**: it will display a simple help message
    - **`@bot_name model-info`**: it will display the information of the currently selected(deployed) model from the [`model_cards.json`](https://github.com/deep-diver/LLM-As-Chatbot/blob/main/model_cards.json).
    - **`@bot_name default-params`**: it will display the default parameters to be used in model's `generate` method. That is `GenerationConfig`, and it holds parameters such as `temperature`, `top_p`, and so on.
    - **`@bot_name user message --max-new-tokens 512 --temperature 0.9 --top-p 0.75 --do_sample --max-windows 5 --internet`**: all parameters are used to dynamically determine the text geneartion behaviour as in `GenerationConfig` except `max-windows`. The `max-windows` determines how many past conversations to look up as a reference. The default value is set to `3`, but as the conversation goes long, you can increase this value. `--internet` will try to answer to your prompt by aggregating information scraped from google search. To use `--internet` option, you need to specify `--serper-api-key` when booting up the program.

### Context management

Different model might have different strategies to manage context, so if you want to know the exact strategies applied to each model, take a look at the [`chats`](https://github.com/deep-diver/LLM-As-Chatbot/tree/main/chats) directory. However, here are the basic ideas that I have come up with initially. I have found long prompts will slow down the generation process a lot eventually, so I thought the prompts should be kept as short as possible while as concise as possible at the same time. In the previous version, I have accumulated all the past conversations, and that didn't go well.

- In every turn of the conversation, the past `N` conversations will be kept. Think about the `N` as a hyper-parameter. As an experiment, currently the past 2-3 conversations are only kept for all models.

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
  - [LLMs/Vicuna-7b-v1.3](https://huggingface.co/lmsys/vicuna-7b-v1.3)
  - [LLMs/Vicuna-13b-v1.1](https://huggingface.co/LLMs/Vicuna-13b-v1.1): Vicuna(13B) from FastChat. This is not a delta weight, so use it at your own risk. I will make this repo as private soon and add Hugging Face token field.
  - [LLMs/Vicuna-13b-v1.3](https://huggingface.co/lmsys/vicuna-13b-v1.3)
  - [LLMs/Vicuna-33b-v1.3](https://huggingface.co/lmsys/vicuna-33b-v1.3)
  - [togethercomputer/RedPajama-INCITE-Chat-7B-v0.1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-7B-v0.1): RedPajama INCITE Chat(7B) from Together.
  - [mosaicml/mpt-7b-chat](https://huggingface.co/mosaicml/mpt-7b-chat): MPT-7B from MOSAIC ML.
  - [mosaicml/mpt-30b-chat](https://huggingface.co/mosaicml/mpt-30b-chat): MPT-30B from MOSAIC ML.
  - [teknium/llama-deus-7b-v3-lora](https://huggingface.co/teknium/llama-deus-7b-v3-lora): LLaMA 7B based Alpaca style instruction fine-tuned model. The only difference between Alpaca is that this model is fine-tuned on more data including Alpaca dataset, GPTeacher, General Instruct, Code Instruct, Roleplay Instruct, Roleplay V2 Instruct, GPT4-LLM Uncensored, Unnatural Instructions, WizardLM Uncensored, CamelAI's 20k Biology, 20k Physics, 20k Chemistry, 50k Math GPT4 Datasets, and CodeAlpaca
  - [HuggingFaceH4/starchat-alpha](https://huggingface.co/HuggingFaceH4/starchat-alpha): Starcoder 15.5B based instruction fine-tuned model. This model is particularly good at answering questions about coding. 
  - [HuggingFaceH4/starchat-beta](https://huggingface.co/HuggingFaceH4/starchat-beta): Starcoder 15.5B based instruction fine-tuned model. This model is particularly good at answering questions about coding.
  - [LLMs/Vicuna-LoRA-EvolInstruct-7B](https://huggingface.co/LLMs/Vicuna-LoRA-EvolInstruct-7B): LLaMA 7B based Vicuna style instruction fine-tuned model. The dataset to fine-tune this model is from WizardLM's Evol Instruction dataset.
  - [LLMs/Vicuna-LoRA-EvolInstruct-13B](https://huggingface.co/LLMs/Vicuna-LoRA-EvolInstruct-13B): LLaMA 13B based Vicuna style instruction fine-tuned model. The dataset to fine-tune this model is from WizardLM's Evol Instruction dataset.
  - [project-baize/baize-v2-7b](https://huggingface.co/project-baize/baize-v2-7b): LLaMA 7B based Baize
  - [project-baize/baize-v2-13b](https://huggingface.co/project-baize/baize-v2-7b): LLaMA 13B based Baize
  - [timdettmers/guanaco-7b](https://huggingface.co/timdettmers/guanaco-7b): LLaMA 7B based Guanaco which is fine-tuned on OASST1 dataset with QLoRA techniques introduced in "QLoRA: Efficient Finetuning of Quantized LLMs" paper. 
  - [timdettmers/guanaco-13b](https://huggingface.co/timdettmers/guanaco-13b): LLaMA 13B based Guanaco which is fine-tuned on OASST1 dataset with QLoRA techniques introduced in "QLoRA: Efficient Finetuning of Quantized LLMs" paper.
  - [timdettmers/guanaco-33b-merged](https://huggingface.co/timdettmers/guanaco-33b-merged): LLaMA 30B based Guanaco which is fine-tuned on OASST1 dataset with QLoRA techniques introduced in "QLoRA: Efficient Finetuning of Quantized LLMs" paper.
  - [tiiuae/falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct): Falcon 7B based instruction fine-tuned model on Baize, GPT4All, GPTeacher, and RefinedWeb-English datasets.
  - [tiiuae/falcon-40b-instruct](https://huggingface.co/tiiuae/falcon-40b-instruct): Falcon 40B based instruction fine-tuned model on Baize and RefinedWeb-English datasets.
  - [LLMs/WizardLM-13B-V1.0](https://huggingface.co/LLMs/WizardLM-13B-V1.0)
  - [LLMs/WizardLM-30B-V1.0](https://huggingface.co/LLMs/WizardLM-30B-V1.0)
  - [ehartford/Wizard-Vicuna-13B-Uncensored](https://huggingface.co/ehartford/Wizard-Vicuna-13B-Uncensored)
  - [ehartford/Wizard-Vicuna-30B-Uncensored](https://huggingface.co/ehartford/Wizard-Vicuna-30B-Uncensored)
  - [ehartford/samantha-7b](https://huggingface.co/ehartford/samantha-7b)
  - [ehartford/samantha-13b](https://huggingface.co/ehartford/samantha-13b)
  - [ehartford/samantha-33b](https://huggingface.co/ehartford/samantha-33b)
  - [CalderaAI/30B-Lazarus](https://huggingface.co/CalderaAI/30B-Lazarus)
  - [elinas/chronos-13b](https://huggingface.co/elinas/chronos-13b)
  - [elinas/chronos-33b](https://huggingface.co/elinas/chronos-33b)
  - [WizardLM/WizardCoder-15B-V1.0](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0)
  - [ehartford/WizardLM-Uncensored-Falcon-7b](https://huggingface.co/ehartford/WizardLM-Uncensored-Falcon-7b)
  - [ehartford/WizardLM-Uncensored-Falcon-40b](https://huggingface.co/ehartford/WizardLM-Uncensored-Falcon-40b)

</details>

## Todos

- [X] Gradio components to control the configurations of the generation
- [X] Multiple conversation management
- [X] Internet search capability (by integrating ChromaDB, `intfloat/e5-large-v2`)
- [ ] Implement server only option w/ FastAPI

## Acknowledgements

- I am thankful to [Jarvislabs.ai](https://jarvislabs.ai/) who generously provided free GPU resources to experiment with Alpaca-LoRA deployment and share it to communities to try out.
- I am thankful to [AI Network](https://www.ainetwork.ai) who generously provided A100(40G) x 8 DGX workstation for fine-tuning and serving the models.
