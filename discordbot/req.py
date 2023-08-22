import re
import copy
import global_vars

from discordbot.utils import (
    get_chat_manager,
    get_global_context
)
from discordbot.flags import (
    parse_known_flags,
    known_flags_def
)

from pingpong import PingPong
from pingpong.context import CtxLastWindowStrategy

from discord import NotFound

from transformers import GenerationConfig
from text_generation import Client

async def tgi_gen(prompt, args):
    gen_config = copy.deepcopy(global_vars.gen_config)
    if args["max-new-tokens"] is not None:        
        gen_config.max_new_tokens = args["max-new-tokens"]
    if args["temperature"] is not None:        
        gen_config.temperature = args["temperature"]
    if args["do-sample"] is not None:        
        gen_config.do_sample = args["do-sample"]
    if args["top-p"] is not None:        
        gen_config.top_p = args["top-p"]
    
    client = Client(
        f"http://{global_vars.remote_addr}:{global_vars.remote_port}",
        timeout=100
    )
    
    response = client.generate(
        prompt, 
        do_sample=gen_config.do_sample,
        max_new_tokens=512,
        repetition_penalty=gen_config.repetition_penalty,
        temperature=gen_config.repetition_penalty,
        top_k=gen_config.top_k,
        top_p=gen_config.top_p
    )
    return response.generated_text


def vanilla_gen(prompt, args):
    input_ids = global_vars.tokenizer(prompt, return_tensors="pt").input_ids.to(global_vars.device)
    
    gen_config = copy.deepcopy(global_vars.gen_config)
    if args["max-new-tokens"] is not None:        
        gen_config.max_new_tokens = args["max-new-tokens"]
    if args["temperature"] is not None:        
        gen_config.temperature = args["temperature"]
    if args["do-sample"] is not None:        
        gen_config.do_sample = args["do-sample"]
    if args["top-p"] is not None:        
        gen_config.top_p = args["top-p"]
    
    generated_ids = global_vars.model.generate(
        input_ids=input_ids, 
        generation_config=gen_config
    )
    response = global_vars.tokenizer.decode(generated_ids[0][input_ids.shape[-1]:])
    return response

async def build_prompt(ppmanager, ctx_include=True, win_size=3):
    dummy_ppm = copy.deepcopy(ppmanager)
    if ctx_include:
        dummy_ppm.ctx = get_global_context(global_vars.model_type)
    else:
        dummy_ppm.ctx = ""
    
    lws = CtxLastWindowStrategy(win_size)
    return lws(dummy_ppm)

async def build_ppm(msg, msg_content, username, user_id):
    ppm = get_chat_manager(global_vars.model_type)
    
    channel = msg.channel
    user_msg = msg_content
    
    packs = []    
    partial_count = 0
    total_count = 0
    
    while True:
        try:
            if msg.reference is not None:
                ref_id = msg.reference.message_id
                msg = await channel.fetch_message(ref_id)
                msg_content = msg.content.replace(f"@{username} ", "").replace(f"<@{user_id}> ", "")
                try: 
                    idx = msg_content.index("💬")
                    msg_content = msg_content[idx+1:].strip()
                except:
                    msg_content = msg_content.strip()
                
                msg_content, _ = parse_known_flags(
                    msg_content, 
                    known_flags_def,
                    global_vars.gen_config
                )
                print(msg_content)
                
                packs.insert(
                    0, msg_content
                )
               
                partial_count = partial_count + 1
                if partial_count >= 2:
                    partial_count = 0
            else:
                break
        
        except NotFound:
            break
    
    for idx in range(0, len(packs), 2):
        ppm.add_pingpong(
            PingPong(packs[idx], packs[idx+1])
        )
        
    ppm.add_pingpong(
        PingPong(user_msg, "")
    )
    print(ppm.pingpongs)
    
    return ppm
