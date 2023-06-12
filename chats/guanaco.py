import torch
from transformers import StoppingCriteria, StoppingCriteriaList

import copy
import json
import global_vars
from chats import pre, post
from pingpong import PingPong
from gens.batch_gen import get_output_batch

from pingpong.context import CtxLastWindowStrategy

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_token_ids = [0]
        
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

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
    
    for new_text in streamer:
        if count == 0:
            ppmanager.append_pong(f"![]({global_vars.model_thumbnail_tiny})***[{global_vars.model_type}]***\n")
            count = count + 1
            
        ppmanager.append_pong(new_text)
        yield ppmanager, ppmanager.build_uis()
                
    yield ppmanager, ppmanager.build_uis()

def summarize(
    ppmanager, prompt_to_summarize, win_size,
    temperature, top_p, top_k, repetition_penalty, max_new_tokens,
    num_beams, use_cache, do_sample, eos_token_id, pad_token_id    
):
    ctx = ppmanager.ctx
    last_pong = ppmanager.pingpongs[-1].pong
    ppmanager.add_pingpong(PingPong(prompt_to_summarize, ""))
    prompt = ppmanager.build_prompts(from_idx=-win_size)

    _, gen_config_summarization = pre.build_gen_config(
        temperature, top_p, top_k, repetition_penalty, max_new_tokens,
        num_beams, use_cache, do_sample, eos_token_id, pad_token_id
    )
    summarize_output = get_output_batch(
        global_vars.model, global_vars.tokenizer, [prompt], gen_config_summarization
    )[0].split(prompt_to_summarize)[-1].strip()
    ppmanager.ctx = summarize_output
    ppmanager.pop_pingpong()
    return ppmanager

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
        StoppingCriteriaList([StopOnTokens()]), False
    )
    pre.start_gen(gen_kwargs)

    # handling stream
    for ppmanager, uis in text_stream(ppm, streamer):
        yield "", uis, prompt, str(res)

    ppm = post.strip_pong(ppm)
    yield "", ppm.build_uis(), prompt, str(res)
    
    # summarization
    # ppm.add_pingpong(
    #     PingPong(None, "![](https://i.postimg.cc/ZKNKDPBd/Vanilla-1s-209px.gif)")
    # )
    # yield "", ppm.build_uis(), prompt, state
    # ppm.pop_pingpong()
    
    # ppm = summarize(
    #     ppm, ctx_sum_prompt, ctx_num_lconv,
    #     sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, 
    #     sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid
    # )
    yield "", ppm.build_uis(), prompt, str(res)    