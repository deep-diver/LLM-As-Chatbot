import os
import copy
import json
import types
import asyncio
import argparse
from urllib.request import urlopen
from concurrent.futures import ThreadPoolExecutor

import discord

import global_vars

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
        help_msg = f"""{global_vars.gen_config}""" 
        await msg.channel.send(help_msg)    
    else:
        if err_msg is None:
            ppm = await build_ppm(msg, user_msg, user_name, user_id)

            prompt = await build_prompt(ppm, user_args.max_windows)
            response = await loop.run_in_executor(
                executor, sync_task, 
                prompt, user_args
            )
            if response.endswith("</s>"):
                response = response[:-len("</s>")]
            
            if response.endswith("<|endoftext|>"):
                response = response[:-len("<|endoftext|>")]
                
            response = f"**{model_name}** 💬\n{response.strip()}"
            await msg.reply(response, mention_author=False)
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
        
def main(args):
    if args.token is None:
        args.token = os.getenv('DISCORD_BOT_TOKEN')
        
    if args.model_name is None:
        args.model_name = os.genenv('DISCORD_BOT_MODEL_NAME')
        
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
    max_workers = args.max_workers
    model_name = args.model_name
    
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
    args = parser.parse_args()
    
    main(args)
