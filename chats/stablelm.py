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
    
    prompt = lws(dummy_ppm)  
    return prompt

def text_stream(ppmanager, streamer):
    for new_text in streamer:
        ppmanager.append_pong(new_text)
        yield ppmanager, ppmanager.build_uis()
                
    yield ppmanager, ppmanager.build_uis()

def summarize(ppmanager):
    ctx = ppmanager.ctx
    last_pong = ppmanager.pingpongs[-1].pong
    ping = f'what have we discussed about so far?'
    ppmanager.add_pingpong(PingPong(ping, ""))
    prompt = ppmanager.build_prompts(from_idx=-3)

    print(prompt)
    
    summarize_output = get_output_batch(
        global_vars.model, global_vars.tokenizer, [prompt], global_vars.gen_config_summarization
    )[0].split("what have we discussed about so far?")[-1].strip()
    print("---------------")
    print(summarize_output)
    ppmanager.ctx = summarize_output
    ppmanager.pop_pingpong()
    return ppmanager

def chat_stream(user_message, state):
    ppm = state["ppmanager"]

    # add_ping returns a prompt structured in Alpaca form
    ppm.add_pingpong(
        PingPong(user_message, "")
    )
    prompt = build_prompts(ppm, user_message)
    
    # prepare text generating streamer & start generating
    gen_kwargs, streamer = pre.build(
        prompt, global_vars.gen_config_raw, StoppingCriteriaList([StopOnTokens()])
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
    
    ppm = summarize(ppm)
    state["ppmanager"] = ppm
    yield "", ppm.build_uis(), prompt, state
