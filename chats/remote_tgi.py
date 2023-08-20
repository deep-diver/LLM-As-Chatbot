import json
import requests
import sseclient

async def gen_text(
    prompt, 
    remote_addr,
    remote_port=None,
    remote_token=None,
    parameters=None
):
    if remote_port and remote_port != "":
        remote_addr = f"{remote_addr}:{remote_port}"
        
    headers={
      'Content-type': 'application/json'
    }
    if remote_token is not None:
        headers["Authorization"] = f'Bearer {remote_token}'

    data = {
      'inputs': prompt,
      'stream': True,
      'options': {
          'use_cache': False,
      },
      'parameters': parameters
    }

    r = requests.post(
      remote_addr,
      headers=headers,
      data=json.dumps(data),
      stream=True
    )

    client = sseclient.SSEClient(r)
    for event in client.events():
        yield json.loads(event.data)['token']['text']
    
async def chat_stream(
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
    
    async for result in gen_text(
        prompt, 
        remote_addr=global_vars.remote_addr, 
        remote_port=global_vars.remote_port, 
        remote_token=global_vars.remote_token,
        parameters={
            'max_new_tokens': res_mnts,
            'do_sample': res_sample,
            'return_full_text': False,
            'temperature': res_temp,
            'top_k': res_topk,
            # 'top_p": res_topp
            'repetition_penalty': res_rpen           
        }
    ):
        ppm.append_pong(result)
        yield "", ppm.build_uis(), prompt, str(res)

    ppm = post.strip_pong(ppm)
    yield "", ppm.build_uis(), prompt, str(res)    