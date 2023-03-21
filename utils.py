import re
import yaml

from transformers import GenerationConfig
from strings import SPECIAL_STRS

def get_generation_config(path):
    with open(path, 'rb') as f:
        generation_config = yaml.safe_load(f.read())

    return GenerationConfig(**generation_config["generation_config"])

def generate_prompt(prompt, histories, ctx=None):
    convs = f"""Below is a history of instructions that describe tasks, paired with an input that provides further context. Write a response that appropriately completes the request by remembering the conversation history.
    
"""
    if ctx is not None:
        convs = f"""{ctx}

"""
    
    start_idx = 0
    
    for idx, history in enumerate(histories):
        history_prompt = history[0]
        if history_prompt == SPECIAL_STRS["summarize"]:
            start_idx = idx

    # drop the previous conversations if user has summarized
    for history in histories[start_idx if start_idx == 0 else start_idx+1:]:
        history_prompt = history[0]
        history_response = history[1]
        
        history_response = history_response.replace("<br>", "\n")

        tag_pattern = re.compile(r'<.*?>')
        history_response = re.sub(tag_pattern, '', history_response)

        convs = convs + f"""### Instruction:{history_prompt}

### Response:{history_response}

"""

    convs = convs + f"""### Instruction:{prompt}

### Response:"""

    print(convs)
    return convs

def post_process_stream(bot_response):
    # sometimes model spits out text containing 
    # "### Response:" and "### Instruction:"
    # in this case, we want to stop generating
    if "### Response:" in bot_response or "### Instruction:" in bot_response:
        bot_response = bot_response.replace("### Response:", '').replace("### Instruction:", '').strip()
        return bot_response, True
    
    bot_response = bot_response.replace("\n", "<br>")
    
    multi_space_pattern = r"(  )"    
    replacement_for_multi_space = r'<span class="chat_wrap_space">  <span>'
    
    bot_response = re.sub(multi_space_pattern, replacement_for_multi_space, bot_response)
    return bot_response, False

def post_process_batch(bot_response):
    bot_response = bot_response.split("### Response:")[-1].strip()
    bot_response = bot_response.replace("\n", "<br>")
    
    pattern = r"(  )"
    replacement = r'<span class="chat_wrap_space">  <span>'
    return re.sub(pattern, replacement, bot_response)

def post_processes_batch(bot_responses):
    return [post_process_batch(r) for r in bot_responses]
