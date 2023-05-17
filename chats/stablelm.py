import torch
from transformers import StoppingCriteria, StoppingCriteriaList

import copy
import global_vars
from chats import pre, post
from pingpong import PingPong
from gens.batch_gen import get_output_batch

from pingpong.context import CtxLastWindowStrategy

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def build_prompts(ppmanager, user_message, win_size=3):
    dummy_ppm = copy.deepcopy(ppmanager)
    lws = CtxLastWindowStrategy(win_size)
    
    dummy_ppm.ctx = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

    prompt = lws(dummy_ppm)  
    return prompt

def text_stream(ppmanager, streamer):
    for new_text in streamer:
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
    user_message, state,
    ctx_num_lconv, ctx_sum_prompt,
    res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
    sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid
):
    ppm = state["ppmanager"]

    # add_ping returns a prompt structured in Alpaca form
    ppm.add_pingpong(
        PingPong(user_message, "")
    )
    prompt = build_prompts(ppm, user_message, ctx_num_lconv)
    
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
        ppm = ppmanager
        yield "", uis, prompt, state

    ppm = post.strip_pong(ppm)
    yield "", ppm.build_uis(), prompt, state
    
    # summarization
    ppm.add_pingpong(
        PingPong(None, "![](https://i.postimg.cc/ZKNKDPBd/Vanilla-1s-209px.gif)")
    )
    yield "", ppm.build_uis(), prompt, state
    ppm.pop_pingpong()
    
    ppm = summarize(
        ppm, ctx_sum_prompt, ctx_num_lconv,
        sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, 
        sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid
    )
    state["ppmanager"] = ppm
    yield "", ppm.build_uis(), prompt, state
