import re
import yaml

from transformers import GenerationConfig
from strings import SPECIAL_STRS

def get_generation_config(path):
    with open('generation_config.yaml', 'rb') as f:
        generation_config = yaml.safe_load(f.read())

    return GenerationConfig(**generation_config["generation_config"])

def generate_prompt(prompt, histories, ctx=None):
    print("----inside")
    print(f"ctx:{ctx}")

    ctx = "" if ctx is None or ctx == "" else f"""
    
    Context:{ctx
    
    }"""

    convs = f"""Below is a history of instructions that describe tasks, paired with an input that provides further context. Write a response that appropriately completes the request by remembering the conversation history.

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

        pattern = re.compile(r'<.*?>')
        history_response = re.sub(pattern, '', history_response)        

        convs = convs + f"""### Instruction:{history_prompt}

### Response:{history_response}

"""

    convs = convs + f"""### Instruction:{prompt}

### Response:"""

    print(convs)
    return convs

def post_process(bot_response):
    bot_response = bot_response.split("### Response:")[-1].strip()
    bot_response = bot_response.replace("\n", "<br>")     # .replace(" ", "&nbsp;")
    
    pattern = r"(  )"
    replacement = r'<span class="chat_wrap_space">  <span>'
    return re.sub(pattern, replacement, bot_response)

def post_processes(bot_responses):
    return [post_process(r) for r in bot_responses]
