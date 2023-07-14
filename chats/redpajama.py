import torch
from transformers import StoppingCriteria, StoppingCriteriaList

import copy
import json
import global_vars
from chats import pre, post
from pingpong import PingPong
from gens.batch_gen import get_output_batch

from chats.utils import build_prompts, text_stream, internet_search

class StopOnTokens(StoppingCriteria):
    # ref: https://github.com/togethercomputer/OpenChatKit/blob/7a931c7d7cf3602c93e00db6e27bdc09d3b5f70f/inference/bot.py
    def __init__(self, tokenizer, stop_words, stream_callback):
        super().__init__()
        self._tokenizer = tokenizer
        self._stop_words = stop_words
        self._partial_result = ''
        self._stream_buffer = ''
        self._stream_callback = stream_callback
             
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        first = not self._partial_result
        text = self._tokenizer.decode(input_ids[0, -1])
        self._partial_result += text
        for stop_word in self._stop_words:
            if stop_word in self._partial_result:
                return True
        return False             

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

    if ppm.pingpongs[-1].pong.endswith(":"):
        ppm.pingpongs[-1].pong = ppm.pingpongs[-1].pong[:-1]
        
    ppm = post.strip_pong(ppm)
    yield "", ppm.build_uis(), prompt, str(res)