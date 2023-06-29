import torch
from transformers import StoppingCriteria, StoppingCriteriaList

import re
import copy
import json
import global_vars
from chats import pre, post
from pingpong import PingPong
from gens.batch_gen import get_output_batch

from pingpong.context import CtxLastWindowStrategy

def build_prompts(ppmanager, user_message, global_context, win_size=3):
    dummy_ppm = copy.deepcopy(ppmanager)
    
    dummy_ppm.ctx = global_context
    for pingpong in dummy_ppm.pingpongs:
        pong = pingpong.pong
        first_sentence = pong.split("\n")[0]
        if first_sentence != "" and \
            pre.contains_image_markdown(first_sentence):
            pong = ' '.join(pong.split("\n")[1:]).strip()
            pingpong.pong = pong
            
    lws = CtxLastWindowStrategy(win_size)
    
    prompt = lws(dummy_ppm)
    return prompt

def text_stream(ppmanager, streamer):
    count = 0
    dummy_ppm = copy.deepcopy(ppmanager)
    
    for new_text in streamer:
        if count == 0:
            ppmanager.append_pong(f"![]({global_vars.model_thumbnail_tiny})***[{global_vars.model_type}]***\n")
            dummy_ppm.append_pong(f"![]({global_vars.model_thumbnail_tiny})***[{global_vars.model_type}]***\n")
            count = count + 1
            
        ppmanager.append_pong(new_text)
        dummy_ppm.append_pong(new_text)
        
        if "Assistant: " in ppmanager.pingpongs[-1].pong:
            dummy_ppm.replace_last_pong(
                dummy_ppm.pingpongs[-1].pong.replace("Assistant: ", "")
            )
        
        if "<|endoftext|>" in ppmanager.pingpongs[-1].pong:
            ppmanager.replace_last_pong(
                re.sub(r'[\s|\n].*<\|endoftext\|>.*[\s|\n]', ' ', ppmanager.pingpongs[-1].pong)
            )
            dummy_ppm.replace_last_pong(
                re.sub(r'[\s|\n].*<\|endoftext\|>.*[\s|\n]', ' ', dummy_ppm.pingpongs[-1].pong)
            )            
            break
        
        yield ppmanager, dummy_ppm.build_uis()
                
    yield ppmanager, dummy_ppm.build_uis()

def chat_stream(
    idx, local_data, user_message, state,
    global_context, ctx_num_lconv, ctx_sum_prompt,
    res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
    sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid
):
    res = [
      state["ppmanager_type"].from_json(json.dumps(ppm))
      for ppm in local_data
    ]

    ppm = res[idx]

    # add_ping returns a prompt structured in Alpaca form
    ppm.add_pingpong(
        PingPong(user_message, "")
    )
    prompt = build_prompts(ppm, user_message, global_context, ctx_num_lconv)

    # prepare text generating streamer & start generating
    gen_kwargs, streamer = pre.build(
        prompt,
        res_temp, res_topp, res_topk, res_rpen, res_mnts, 
        res_beams, res_cache, res_sample, res_eosid, res_padid,
        None, False
    )
    pre.start_gen(gen_kwargs)

    # handling stream
    for ppmanager, uis in text_stream(ppm, streamer):
        yield "", uis, prompt, str(res)

    ppm = post.strip_pong(ppm)
    dummy_ppm = copy.deepcopy(ppm)
    
    if "Assistant: " in dummy_ppm.pingpongs[-1].pong:
        dummy_ppm.replace_last_pong(
            dummy_ppm.pingpongs[-1].pong.replace("Assistant: ", "")
        )    
    
    yield "", dummy_ppm.build_uis(), prompt, str(res)