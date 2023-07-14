import copy
import json
import global_vars
from chats import pre, post
from pingpong import PingPong
from gens.batch_gen import get_output_batch

from chats.utils import build_prompts, text_stream, internet_search

def chat_stream(
    idx, local_data, user_message, state,
    global_context, ctx_num_lconv, ctx_sum_prompt,
    res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
    sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid,
    internet_option, serper_api_key
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
    prompt = build_prompts(ppm, global_context, ctx_num_lconv)

    #######
    if internet_option:
        search_prompt = None
        for tmp_prompt, uis in internet_search(ppm, serper_api_key, global_context, ctx_num_lconv):
            search_prompt = tmp_prompt
            yield "", uis, prompt, str(res)
    
    # prepare text generating streamer & start generating
    gen_kwargs, streamer = pre.build(
        search_prompt if internet_option else prompt,
        res_temp, res_topp, res_topk, res_rpen, res_mnts, 
        res_beams, res_cache, res_sample, res_eosid, res_padid,
        return_token_type_ids=False
    )
    pre.start_gen(gen_kwargs)

    # handling stream
    for ppmanager, uis in text_stream(ppm, streamer):
        yield "", uis, prompt, str(res)

    ppm = post.strip_pong(ppm)
    yield "", ppm.build_uis(), prompt, str(res)