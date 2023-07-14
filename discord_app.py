import os
import copy
import json
import types
import asyncio
import argparse
from urllib.request import urlopen
from concurrent.futures import ThreadPoolExecutor

import discord
from discord.errors import HTTPException

import global_vars
from pingpong.context import InternetSearchStrategy, SimilaritySearcher

from discordbot.req import (
    sync_task, build_prompt, build_ppm
)
from discordbot.flags import parse_req

model_info = json.load(open("model_cards.json"))
    
intents = discord.Intents.default()
intents.members = True
client = discord.Client(intents=intents)
queue = asyncio.Queue()

special_words = [
    "help",
    "model-info",
    "default-params",
]
max_response_length = 2000

async def build_prompt_and_reply(executor, user_name, user_id):
    loop = asyncio.get_running_loop()
    
    print(queue.qsize())
    msg = await queue.get()
    user_msg, user_args, err_msg = parse_req(
        msg.content.replace(f"@{user_name} ", "").replace(f"<@{user_id}> ", ""), None
    )
    
    if user_msg == "help":
        help_msg = """Type one of the following for more information about this chatbot
- **`help`:** list of supported commands
- **`model-info`:** get currently selected model card
- **`default-params`:** get default parameters of the Generation Config

You can start conversation by metioning the chatbot `@{chatbot name} {your prompt} {options}`, and the following options are supported.
- **`--top-p {float}`**: determins how many tokens to pick from the top tokens based on the sum of their probabilities(<= `top-p`).
- **`--temperature {float}`**: used to modulate the next token probabilities.
- **`--max-new-tokens {integer}`**: maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
- **`--do-sample`**: determines whether or not to use sampling ; use greedy decoding otherwise.
- **`--max-windows {integer}`**: determines how many past conversations to look up as a reference.
- **`--internet`**: determines whether or not to use internet search capabilities.

If you want to continue conversation based on past conversation histories, you can simply `reply` to chatbot's message. At this time, you don't need to metion its name. However, you need to specify options in every turn. For instance, if you want to `reply` based on internet search information, then you shoul specify `--internet` in your message.
"""
        await msg.channel.send(help_msg)
    elif user_msg == "model-info":
        selected_model_info = model_info[model_name]
        help_msg = f"""## {model_name}
- **Description:** {selected_model_info['desc']}
- **Number of parameters:** {selected_model_info['parameters']}
- **Hugging Face Hub (base):** {selected_model_info['hub(base)']}
- **Hugging Face Hub (ckpt):** {selected_model_info['hub(ckpt)']}
"""        
        await msg.channel.send(help_msg)
    elif user_msg == "default-params":
        help_msg = f"""{global_vars.gen_config}, max-windows = {user_args.max_windows}"""
        await msg.channel.send(help_msg)    
    else:
        if err_msg is None:
            try:
                ppm = await build_ppm(msg, user_msg, user_name, user_id)

                if user_args.internet and serper_api_key is not None:
                    progress_msg = await msg.reply("Progress 🚧", mention_author=False)

                    internet_search_ppm = copy.deepcopy(ppm)
                    internet_search_prompt = f"My question is '{user_msg}'. Based on the conversation history, give me an appropriate query to answer my question for google search. You should not say more than query. You should not say any words except the query."
                    internet_search_ppm.pingpongs[-1].ping = internet_search_prompt
                    internet_search_prompt = await build_prompt(
                        internet_search_ppm, 
                        ctx_include=False,
                        win_size=user_args.max_windows
                    )
                    internet_search_prompt_response = await loop.run_in_executor(
                        executor, sync_task, internet_search_prompt, user_args
                    )
                    if internet_search_prompt_response.endswith("</s>"):
                        internet_search_prompt_response = internet_search_prompt_response[:-len("</s>")]
                    if internet_search_prompt_response.endswith("<|endoftext|>"):
                        internet_search_prompt_response = internet_search_prompt_response[:-len("<|endoftext|>")]

                    ppm.pingpongs[-1].ping = internet_search_prompt_response

                    await progress_msg.edit(
                        content=f"• Search query re-organized by LLM: {internet_search_prompt_response}", 
                        suppress=True
                    )

                    searcher = SimilaritySearcher.from_pretrained(device="cuda")

                    logs = ""
                    for step_ppm, step_msg in InternetSearchStrategy(
                        searcher, serper_api_key=serper_api_key
                    )(ppm, search_query=internet_search_prompt_response, top_k=8):
                        ppm = step_ppm
                        logs = logs + step_msg + "\n"
                        await progress_msg.edit(content=logs, suppress=True)

                prompt = await build_prompt(ppm, win_size=user_args.max_windows)
                response = await loop.run_in_executor(
                    executor, sync_task, 
                    prompt, user_args
                )
                if response.endswith("</s>"):
                    response = response[:-len("</s>")]

                if response.endswith("<|endoftext|>"):
                    response = response[:-len("<|endoftext|>")]

                response = f"**{model_name}** 💬\n{response.strip()}"
                if len(response) >= max_response_length:
                    response = response[:max_response_length]

                if user_args.internet and serper_api_key is not None:
                    await progress_msg.delete()
                
                await msg.reply(response, mention_author=False)
            except IndexError:
                err_msg = "Index error"
                await msg.channel.send(err_msg)
            except HTTPException:
                pass
        else:
            await msg.channel.send(err_msg)
    
async def background_task(user_name, user_id, max_workers):
    executor = ThreadPoolExecutor(max_workers=max_workers)
    print("Task Started. Waiting for inputs.")
    while True:
        # await asyncio.sleep(5)
        await build_prompt_and_reply(executor, user_name, user_id)

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    asyncio.get_running_loop().create_task(
        background_task(
            client.user.name,
            client.user.id,
            max_workers,
        )
    )

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if isinstance(message.channel, discord.channel.DMChannel) or\
        (client.user and client.user.mentioned_in(message)):
        await queue.put(message)

def off_modes(args):
    args.mode_cpu = False
    args.mode_mps = False
    args.mode_8bit = False
    args.mode_4bit = False
    args.mode_full_gpu = False
    return args
        
def discord_main(args):
    if args.token is None:
        args.token = os.getenv('DISCORD_BOT_TOKEN')
        
    if args.model_name is None:
        args.model_name = os.getenv('DISCORD_BOT_MODEL_NAME')
        
    if args.token is None or args.model_name is None:
        print('Either or both of token and model-name is not provided')
        print('Set them through CLI or environment variables(DISCORD_BOT_TOKEN, DISCORD_BOT_MODEL_NAME)')
        quit()

    if os.getenv('DISCORD_BOT_MAX_WORKERS'):
        args.max_workers = int(os.getenv('DISCORD_BOT_MAX_WORKERS'))
        
    if os.getenv('DISCORD_BOT_LOAD_MODE'):
        mode = os.getenv('DISCORD_BOT_LOAD_MODE')
        
        if mode == "CPU":
            off_modes(args)
            args.mode_cpu = True
        elif mode == "MPS":
            off_modes(args)
            args.mode_mps = True            
        elif mode == "8BIT":
            off_modes(args)
            args.mode_8bit = True            
        elif mode == "4BIT":
            off_modes(args)
            args.mode_4bit = True            
        elif mode == "HALF":
            off_modes(args)
            args.mode_full_gpu = True            

    global max_workers
    global model_name
    global serper_api_key
    max_workers = args.max_workers
    model_name = args.model_name
    serper_api_key = args.serper_api_key
    
    selected_model_info = model_info[model_name]
    
    tmp_args = types.SimpleNamespace()
    tmp_args.base_url = selected_model_info['hub(base)']
    tmp_args.ft_ckpt_url = selected_model_info['hub(ckpt)']
    tmp_args.gen_config_path = selected_model_info['default_gen_config']
    tmp_args.gen_config_summarization_path = selected_model_info['default_gen_config']
    tmp_args.force_download_ckpt = False
    tmp_args.thumbnail_tiny = selected_model_info['thumb-tiny']
    
    tmp_args.mode_cpu = args.mode_cpu
    tmp_args.mode_mps = args.mode_mps
    tmp_args.mode_8bit = args.mode_8bit
    tmp_args.mode_4bit = args.mode_4bit
    tmp_args.mode_full_gpu = args.mode_full_gpu
    tmp_args.local_files_only = args.local_files_only
    
    try:
        global_vars.initialize_globals(tmp_args)
    except RuntimeError as e:
        print("GPU memory is not enough to load this model.")
        quit()
    
    client.run(args.token)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    # can be set via environment variable
    # --token == DISCORD_BOT_TOKEN
    # --model-name == DISCORD_BOT_MODEL_NAME
    parser.add_argument('--token', default=None, type=str)
    parser.add_argument('--model-name', default=None, type=str) 
    parser.add_argument('--max-workers', default=1, type=int)
    parser.add_argument('--mode-cpu', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--mode-mps', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--mode-8bit', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--mode-4bit', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--mode-full-gpu', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--local-files-only', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--serper-api-key', default=None, type=str)
    args = parser.parse_args()
    
    discord_main(args)
