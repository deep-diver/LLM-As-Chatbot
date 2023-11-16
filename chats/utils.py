import copy
import global_vars

from pingpong.context import CtxLastWindowStrategy
from pingpong.context import InternetSearchStrategy, SimilaritySearcher

from chats import pre, post

def build_prompts(ppmanager, global_context, win_size=3):
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
            ppmanager.append_pong(f"![]({global_vars.model_thumbnail_tiny})***[{global_vars.model_name}]***\n\n")
            count = count + 1
            
        ppmanager.append_pong(new_text)
        yield ppmanager, ppmanager.build_uis()
                
    yield ppmanager, ppmanager.build_uis()
    
def internet_search(ppmanager, serper_api_key, global_context, ctx_num_lconv, device="cpu"):
    instruction = "Based on the provided texts below, please answer to '{ping}' in your own words. Try to explain in detail as much as possible."
    
    searcher = SimilaritySearcher.from_pretrained(device=device)
    iss = InternetSearchStrategy(
        searcher, 
        instruction=instruction, 
        serper_api_key=serper_api_key
    )(ppmanager)

    step_ppm = None
    while True:
        try:
            step_ppm, _ = next(iss)
            yield "", step_ppm.build_uis()
        except StopIteration:
            break

    search_prompt = build_prompts(step_ppm, global_context, ctx_num_lconv)
    yield search_prompt, ppmanager.build_uis()