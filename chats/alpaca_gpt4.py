import copy
import global_vars
from chats import pre, post
from pingpong import PingPong
from gens.batch_gen import get_output_batch

from pingpong.context import CtxLastWindowStrategy

def build_prompt(ppmanager, user_message, win_size=2):
    dummy_ppm = copy.deepcopy(ppmanager)
    dummy_ppm.pop_pingpong()
    lws = CtxLastWindowStrategy(win_size)
    
    lws_result = lws(dummy_ppm, truncate_size=100)
    lws_result = lws_result.replace("### Instruction:\n", "I said to you as: ")
    lws_result = lws_result.replace("### Response:\n", "You responded back to me as: ")
    if lws_result != "":
        lws_result = f'''Given the recent conversation between you and me of "
-----
{lws_result}
-----"
the global context of this conversations is as follow in your perspective of "'''
    
    if dummy_ppm.ctx:
        dummy_ppm.ctx = lws_result + dummy_ppm.ctx + '"'
    else:
        dummy_ppm.ctx = lws_result + dummy_ppm.ctx
        
    prompts = dummy_ppm.add_ping(user_message)    
    return prompts

def summarize(ppmanager):
    ctx = ppmanager.ctx
    pong = ppmanager.pingpongs[-1].pong
    if ctx is None or ctx == "":
        ping = f'given the context of "{ctx}", summarize your response of "{pong}"'
    else:
        ping = f'summarize "{pong}"'
    prompt = ppmanager.add_ping(ping)
    
    summarize_output = get_output_batch(
        global_vars.model, global_vars.tokenizer, [prompt], global_vars.gen_config_summarization
    )[0].split("### Response:")[-1].strip()
    ppmanager.ctx = summarize_output
    ppmanager.pop_pingpong()
    return ppmanager

def wipe_weird_pong_ends(ppmanager):
    last_pong = ppmanager.pingpongs[-1].pong
    last_pong_len = len(last_pong)
    
    tmp_idx = last_pong_len-1
    for char in reversed(last_pong):
        if char in ["!", ".", "?"] \
            and tmp_idx != last_pong_len-1: 
            last_pong = last_pong[:tmp_idx+1]
            break
            
        tmp_idx -= 1

    ppmanager.pingpongs[-1].pong = last_pong
    return ppmanager
    
def text_stream(ppmanager, streamer):
    sandbox = ""
    sandbox_enabled = False
    for new_text in streamer:
        new_text = new_text.replace("�", "")
        
        if "###" in new_text:
            sandbox_enabled = True
            sandbox = new_text
        elif "Instruction:" in new_text \
            or "Response:" in new_text \
            or "Comment:" in new_text \
            or "Commentary:" in new_text:
            break
        else:
            if sandbox_enabled:
                sandbox += new_text
                sandbox_enabled = False

            if "### Instruction:" in sandbox or \
                "### Response:" in sandbox or \
                "### Input:" in sandbox:
                break
            else:
                ppmanager.append_pong(new_text)
                yield ppmanager, ppmanager.build_uis()
                
    yield ppmanager, ppmanager.build_uis()
    
def chat_stream(user_message, state):
    ppm = state["ppmanager"]

    # add_ping returns a prompt structured in Alpaca form
    # add_pong("") means no response yet (to avoid None). Later, tokens will be appended
    ppm.add_pingpong(
        PingPong(user_message, "")
    )
    prompt = build_prompt(ppm, user_message)
    
    # prepare text generating streamer & start generating
    gen_kwargs, streamer = pre.build(prompt, global_vars.gen_config_raw)
    pre.start_gen(gen_kwargs)

    # handling stream
    for _, uis in text_stream(ppm, streamer):
        yield "", uis, prompt, state

    ppm = post.strip_pong(ppm)
    ppm = wipe_weird_pong_ends(ppm)
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