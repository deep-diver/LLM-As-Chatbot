import re

from miscs.strings import SPECIAL_STRS
from miscs.constants import html_tag_pattern
from miscs.constants import repl_empty_str

def generate_prompt(
    prompt, 
    histories, 
    ctx=None, 
    user_indicator="### Instruction:", 
    ai_indicator="### Response:", 
    partial=False
):
    convs = ""
    if ctx is not None:
        convs = f"""{ctx}

"""
    sub_convs = ""
    start_idx = 0
    
    for idx, history in enumerate(histories):
        history_prompt = history[0]
        history_response = history[1]
        if history_response == "✅ summarization is done and set as context" or history_prompt == SPECIAL_STRS["summarize"]:
            start_idx = idx

    # drop the previous conversations if user has summarized
    for history in histories[start_idx if start_idx == 0 else start_idx+1:]:
        history_prompt = history[0]
        history_response = history[1]
        
        history_response = history_response.replace("<br>", "\n")
        history_response = re.sub(
            html_tag_pattern, repl_empty_str, history_response
        )

        sub_convs = sub_convs + f"""{user_indicator} {history_prompt}

{ai_indicator} {history_response}

"""

    sub_convs = sub_convs + f"""{user_indicator} {prompt}

{ai_indicator} """

    convs = convs + sub_convs
    print(convs)
    return sub_convs if partial else convs, len(sub_convs)
