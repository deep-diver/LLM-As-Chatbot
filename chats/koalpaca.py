import copy
import global_vars
from chats import pre, post
from pingpong import PingPong
from gens.batch_gen import get_output_batch

from pingpong.context import CtxLastWindowStrategy

def build_prompts(ppmanager, user_message, win_size=3):
    dummy_ppm = copy.deepcopy(ppmanager)
    lws = CtxLastWindowStrategy(win_size)
    
    dummy_ppm.ctx = """아래는 인간과 AI 어시스턴트 간의 일련의 대화입니다.
인공지능은 주어진 질문에 대한 응답으로 대답을 시도합니다.
인공지능은 `### 질문` 또는 `### 응답`가 포함된 텍스트를 생성해서는 안 됩니다.
AI는 도움이 되고, 예의 바르고, 정직하고, 정교하고, 감정을 인식하고, 겸손하지만 지식이 있어야 합니다.
어시스턴트는 거의 모든 것을 기꺼이 도와줄 수 있어야 하며, 무엇이 필요한지 정확히 이해하기 위해 최선을 다해야 합니다.
또한 허위 또는 오해의 소지가 있는 정보를 제공하지 않아야 하며, 정답을 완전히 확신할 수 없을 때는 주의를 환기시켜야 합니다.
즉, 이 어시스턴트는 실용적이고 정말 최선을 다하며 주의를 기울이는 데 너무 많은 시간을 할애하지 않습니다.
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
    )[0].split("### 응답:")[-1].strip()
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
        return_token_type_ids=False
    )
    pre.start_gen(gen_kwargs)

    # handling stream
    for ppmanager, uis in text_stream(ppm, streamer):
        yield "", uis, prompt, state

    ppm = post.strip_pong(ppm)
    yield "", ppm.build_uis(), prompt, state
    
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
    state["ppmanager"] = ppm
    yield "", ppm.build_uis(), prompt, state