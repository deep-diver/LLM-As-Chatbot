from chats import stablelm
from chats import alpaca
from chats import koalpaca
from chats import flan_alpaca
from chats import os_stablelm
from chats import vicuna
from chats import stable_vicuna
from chats import starchat
from chats import wizard_coder
from chats import redpajama
from chats import mpt
from chats import alpacoom
from chats import baize
from chats import guanaco
from chats import falcon
from chats import wizard_falcon
from chats import xgen
from chats import llama2
from chats import freewilly
from chats import custom

import copy
import json
import requests
import sseclient
import global_vars
from pingpong import PingPong
from chats import remote_tgi
from chats import pre, post
from chats.utils import build_prompts, text_stream, internet_search

async def chat_stream(
    idx, local_data, user_message, state,
    global_context, ctx_num_lconv, ctx_sum_prompt,
    res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
    sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid,
    internet_option, serper_api_key
):
    if global_vars.remote_addr != "":
        if internet_option == "on" and serper_api_key.strip() != "":       
            internet_option = True
        else:
            internet_option = False        
        
        res = [
          state["ppmanager_type"].from_json(json.dumps(ppm))
          for ppm in local_data
        ]

        ppm = res[idx]

        # add_ping returns a prompt structured in Alpaca form
        ppm.add_pingpong(
            PingPong(user_message, "")
        )
        prompt = build_prompts(ppm, global_context, ctx_num_lconv)

        #######
        print("internet_option", internet_option)
        if internet_option:
            search_prompt = None
            for tmp_prompt, uis in internet_search(ppm, serper_api_key, global_context, ctx_num_lconv):
                search_prompt = tmp_prompt
                yield "", uis, prompt, str(res)
                
        count = 0
                
        async for result in remote_tgi.gen_text(
            prompt, 
            remote_addr=global_vars.remote_addr, 
            remote_port=global_vars.remote_port, 
            remote_token=global_vars.remote_token,
            parameters={
                'max_new_tokens': res_mnts,
                'do_sample': res_sample,
                'return_full_text': False,
                'temperature': res_temp,
                'top_k': res_topk,
                # 'top_p": res_topp
                'repetition_penalty': res_rpen           
            }
        ):
            if count == 0:
                ppm.append_pong(f"![]({global_vars.model_thumbnail_tiny})***[{global_vars.model_name}]***\n")
                count = count + 1
                
            ppm.append_pong(result)
            yield "", ppm.build_uis(), prompt, str(res)

        ppm = post.strip_pong(ppm)
        yield "", ppm.build_uis(), prompt, str(res)
        
    else:
        cs = sync_chat_stream(
            idx, local_data, user_message, state, 
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid,
            internet_option, serper_api_key            
        )
        
        for idx, x in enumerate(cs):
            yield x 

def sync_chat_stream(
    idx, local_data, user_message, state, 
    global_context, ctx_num_lconv, ctx_sum_prompt,
    res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
    sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid,
    internet_option, serper_api_key
):
    model_type = state["model_type"]
    
    if internet_option == "on" and serper_api_key.strip() != "":       
        internet_option = True
    else:
        internet_option = False

    if model_type == "custom":
        cs = custom.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )
    
    elif model_type == "puffin":
        cs = alpaca.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )        
    
    elif model_type == "platypus2":
        cs = alpaca.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )        
    
    elif model_type == "free-willy":
        cs = freewilly.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )        
    
    elif model_type == "upstage-llama" or model_type == "upstage-llama2":
        cs = vicuna.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )
    
    elif model_type == "llama2":
        cs = llama2.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )        
    
    elif model_type == "xgen":
        cs = xgen.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )
    
    elif model_type == "stablelm":
        cs = stablelm.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type == "falcon":
        cs = falcon.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type == "wizard-falcon":
        cs = wizard_falcon.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )
        
    elif model_type == "baize":
        cs = baize.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )        
        
    elif model_type == "alpaca":
        cs = alpaca.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )
        
    elif model_type == "openllama":
        cs = alpaca.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )
        
    elif model_type == "orcamini":
        cs = alpaca.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )        

    elif model_type == "alpaca-gpt4":
        cs = alpaca.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type == "nous-hermes":
        cs = alpaca.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type == "replit-instruct":
        cs = alpaca.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )        

    elif model_type == "alpacoom":
        cs = alpacoom.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )        

    elif model_type == "llama-deus":
        cs = alpaca.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type == "camel":
        cs = alpaca.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type == "koalpaca-polyglot":
        cs = koalpaca.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type == "kullm-polyglot":
        cs = koalpaca.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )        

    elif model_type == "flan-alpaca":
        cs = flan_alpaca.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )        

    elif model_type == "os-stablelm":
        cs = os_stablelm.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type == "t5-vicuna":
        cs = vicuna.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type == "stable-vicuna":
        cs = stable_vicuna.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type == "vicuna":
        cs = vicuna.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )
        
    elif model_type == "wizardlm" or model_type == "wizardlm2":
        cs = vicuna.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )
        
    elif model_type == "wizard-vicuna":
        cs = vicuna.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )        
        
    elif model_type == "airoboros":
        cs = vicuna.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )
        
    elif model_type == "samantha-vicuna":
        cs = vicuna.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )        

    elif model_type == "evolinstruct-vicuna":
        cs = vicuna.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type == "starchat":
        cs = starchat.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )
        
    elif model_type == "wizard-coder":
        cs = wizard_coder.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )
        
    elif model_type == "mpt":
        cs = mpt.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type == "redpajama":
        cs = redpajama.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )
        
    elif model_type == "redpajama-instruct":
        cs = redpajama.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )        

    elif model_type == "guanaco":
        cs = guanaco.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )

    elif model_type == "lazarus":
        cs = alpaca.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )
    
    elif model_type == "chronos":
        cs = alpaca.chat_stream(
            idx, local_data, user_message, state,
            global_context, ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid, 
            internet_option, serper_api_key
        )        
        
    for idx, x in enumerate(cs):
        yield x 
        