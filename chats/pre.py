import copy
import global_vars
from threading import Thread
from transformers import TextIteratorStreamer

def build_model_inputs(prompt):
    model_inputs = global_vars.tokenizer(
        [prompt], 
        return_tensors="pt"
    ).to("cuda")
    return model_inputs

def build_streamer(
    timeout=20.,
    skip_prompt=True,
    skip_special_tokens=True
):
    streamer = TextIteratorStreamer(
        global_vars.tokenizer, 
        timeout=timeout, 
        skip_prompt=skip_prompt,
        skip_special_tokens=skip_special_tokens
    )
    return streamer

def build_gen_kwargs(
    gen_config,
    model_inputs,
    streamer,
    stopping_criteria
):
    gen_kwargs = dict(
        model_inputs,
        streamer=streamer,
        stopping_criteria=stopping_criteria
    )
    gen_kwargs.update(gen_config)
    return gen_kwargs 

def start_gen(gen_kwargs):
    t = Thread(
        target=global_vars.stream_model.generate,
        kwargs=gen_kwargs
    )
    t.start()
    
def build(prompt, gen_config_raw, stopping_criteria=None):
    model_inputs = build_model_inputs(prompt)
    streamer = build_streamer()
    gen_kwargs = build_gen_kwargs(
        gen_config_raw, 
        model_inputs, 
        streamer,
        stopping_criteria
    )
    return gen_kwargs, streamer