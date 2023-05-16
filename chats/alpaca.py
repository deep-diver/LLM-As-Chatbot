import copy
import global_vars
from chats import pre, post
from pingpong import PingPong
from gens.batch_gen import get_output_batch

from pingpong.context import CtxLastWindowStrategy

def build_prompts(ppmanager, user_message, win_size=3):
    dummy_ppm = copy.deepcopy(ppmanager)
    dummy_ppm.ctx = """Below are a series of dialogues between human and an AI assistant.
The AI tries to answer the given instruction as in response.
The AI MUST not generate any text containing `### Response` or `### Instruction`.
The AI MUST be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable.
The assistant MUST be happy to help with almost anything, and will do its best to understand exactly what is needed.
It also MUST avoid giving false or misleading information, and it caveats when it isn’t entirely sure about the right answer.
That said, the assistant is practical and really does its best, and doesn’t let caution get too much in the way of being useful.
"""
    lws = CtxLastWindowStrategy(win_size)
    
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
    )[0].split("### Response:")[-1].strip()
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