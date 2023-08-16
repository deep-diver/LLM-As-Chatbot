import os
import copy
import json
import types
import asyncio
import argparse
from urlextract import URLExtract
from urllib.request import urlopen
from concurrent.futures import ThreadPoolExecutor

import discord
from discord.errors import HTTPException

import global_vars
from pingpong.context import InternetSearchStrategy, SimilaritySearcher

from discordbot.req import (
    tgi_gen, vanilla_gen, build_prompt, build_ppm
)
from discordbot.flags import parse_req
from discordbot import helps, post
from dumb_utils import URLSearchStrategy

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
    other_job_on_progress = False
    loop = asyncio.get_running_loop()
    
    print(queue.qsize())
    msg = await queue.get()
    user_msg, user_args = parse_req(
        msg.content.replace(f"@{user_name} ", "").replace(f"<@{user_id}> ", ""), global_vars.gen_config
    )
    user_args.tgi_server_addr = tgi_server_addr
    user_args.tgi_server_port = tgi_server_port
    
    if user_msg == "help":
        await msg.channel.send(helps.get_help())
    elif user_msg == "model-info":
        await msg.channel.send(helps.get_model_info(model_name, model_info))
    elif user_msg == "default-params":
        await msg.channel.send(helps.get_default_params(global_vars.gen_config, user_args["max-windows"]))
    else:
        try:
            ppm = await build_ppm(msg, user_msg, user_name, user_id)

            if user_args["internet"] and serper_api_key is not None:
                other_job_on_progress = True
                progress_msg = await msg.reply("Progress 🚧", mention_author=False)

                internet_search_ppm = copy.deepcopy(ppm)
                internet_search_prompt = f"My question is '{user_msg}'. Based on the conversation history, give me an appropriate query to answer my question for google search. You should not say more than query. You should not say any words except the query."
                internet_search_ppm.pingpongs[-1].ping = internet_search_prompt
                internet_search_prompt = await build_prompt(
                    internet_search_ppm, 
                    ctx_include=False,
                    win_size=user_args["max-windows"]
                )
                internet_search_prompt_response = await loop.run_in_executor(
                    executor, gen_method, internet_search_prompt, user_args
                )
                internet_search_prompt_response = post.clean(internet_search_prompt_response)

                ppm.pingpongs[-1].ping = internet_search_prompt_response

                await progress_msg.edit(
                    content=f"• Search query re-organized by LLM: {internet_search_prompt_response}", 
                    suppress=True
                )

                searcher = SimilaritySearcher.from_pretrained(device=global_vars.device)

                logs = ""
                for step_ppm, step_msg in InternetSearchStrategy(
                    searcher, serper_api_key=serper_api_key
                )(ppm, search_query=internet_search_prompt_response, top_k=8):
                    ppm = step_ppm
                    logs = logs + step_msg + "\n"
                    await progress_msg.edit(content=logs, suppress=True)
            else:
                url_extractor = URLExtract()
                urls = url_extractor.find_urls(user_msg)
                print(f"urls = {urls}")

                if len(urls) > 0:
                    progress_msg = await msg.reply("Progress 🚧", mention_author=False)

                    other_job_on_progress = True
                    searcher = SimilaritySearcher.from_pretrained(device=global_vars.device)

                    logs = ""
                    for step_result, step_ppm, step_msg in URLSearchStrategy(searcher)(ppm, urls, top_k=8):
                        if step_result is True:
                            ppm = step_ppm
                            logs = logs + step_msg + "\n"
                            await progress_msg.edit(content=logs, suppress=True)
                        else:
                            ppm = step_ppm
                            logs = logs + step_msg + "\n"
                            await progress_msg.edit(content=logs, suppress=True)
                            await asyncio.sleep(2)
                            break

            prompt = await build_prompt(ppm, win_size=user_args["max-windows"])
            response = await loop.run_in_executor(executor, gen_method, prompt, user_args)
            response = post.clean(response)

            response = f"**{model_name}** 💬\n{response.strip()}"
            if len(response) >= max_response_length:
                response = response[:max_response_length]

            if other_job_on_progress is True:
                await progress_msg.delete()

            await msg.reply(response, mention_author=False)
        except IndexError:
            await msg.channel.send("Index error")
        except HTTPException:
            pass
    
async def background_task(user_name, user_id, max_workers):
    executor = ThreadPoolExecutor(max_workers=max_workers)
    print("Task Started. Waiting for inputs.")
    while True:
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

    if os.getenv('TGI_SERVER_ADDR') and os.getenv('TGI_SERVER_PORT'):
        args.tgi_server_addr = os.getenv('TGI_SERVER_ADDR')
        args.tgi_server_port = os.getenv('TGI_SERVER_PORT')

    global max_workers
    global model_name
    global serper_api_key
    global gen_method
    global tgi_server_addr
    global tgi_server_port
    
    max_workers = args.max_workers
    model_name = args.model_name
    serper_api_key = args.serper_api_key
    gen_method = vanilla_gen
    tgi_server_addr = None
    tgi_server_port = None
    
    if args.tgi_server_addr is not None and \
        args.tgi_server_port is not None:
        tgi_server_addr = args.tgi_server_addr
        tgi_server_port = args.tgi_server_port
        
        gen_method = tgi_gen
    
    selected_model_info = model_info[model_name]
    
    tmp_args = types.SimpleNamespace()
    tmp_args.base_url = selected_model_info['hub(base)']
    tmp_args.ft_ckpt_url = selected_model_info['hub(ckpt)']
    tmp_args.gptq_url = None
    tmp_args.gptq_base_url = None
    tmp_args.gen_config_path = selected_model_info['default_gen_config']
    tmp_args.gen_config_summarization_path = selected_model_info['default_gen_config']
    tmp_args.force_download_ckpt = False
    tmp_args.thumbnail_tiny = selected_model_info['thumb-tiny']
    
    tmp_args.mode_cpu = args.mode_cpu
    tmp_args.mode_mps = args.mode_mps
    tmp_args.mode_8bit = args.mode_8bit
    tmp_args.mode_4bit = args.mode_4bit
    tmp_args.mode_full_gpu = args.mode_full_gpu
    tmp_args.mode_gptq = False
    tmp_args.mode_mps_gptq = False
    tmp_args.mode_cpu_gptq = False
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
    parser.add_argument('--tgi-server-addr', default=None, type=str)
    parser.add_argument('--tgi-server-port', default=None, type=str)
    args = parser.parse_args()
    
    discord_main(args)
