from strings import TITLE, ABSTRACT, BOTTOM_LINE

import os
import re
import sys
import argparse
import time
from itertools import chain
from functools import partial
import gradio as gr
import datetime

from model import load_model
from gen import get_output

model, tokenizer = load_model()

def generate_prompt(prompt, ctx=None):
    if ctx.strip() == "":
        ctx = "This is the initial conversation"

    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Input:{ctx}

### Instruction:{prompt}

### Response:"""

def test_fn(prompt, histories, ctx=None):
    print("----inside")

    ctx = "" if ctx is None or ctx == "" else f"""
    
    Context:{ctx
    
    }"""

    convs = f"""Below is a history of instructions that describe tasks, paired with an input that provides further context. Write a response that appropriately completes the request by remembering the conversation history.
{ctx}
"""

    for history in histories:
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

def chat(
    contexts,
    instructions, 
    state_chatbots
):
    print("-------state_chatbots------")
    print(state_chatbots)
    results = []

    # instruct_prompts = [generate_prompt(instruct, ctx) for ctx, instruct in zip(contexts, instructions)]
    instruct_prompts = [test_fn(instruct, histories, ctx) for ctx, instruct, histories in zip(contexts, instructions, state_chatbots)]
        
    bot_responses = get_output(
        model, tokenizer, instruct_prompts, None
    )
    print(bot_responses)
    bot_responses = post_processes(bot_responses)

    print("zipping...")
    sub_results = []
    for instruction, bot_response, state_chatbot in zip(instructions, bot_responses, state_chatbots):
        print(instruction)
        print(bot_response)
        print(state_chatbot)
        new_state_chatbot = state_chatbot + [(instruction, bot_response)]
        print(new_state_chatbot)
        results.append(new_state_chatbot)

    print(results)
    print(len(results))

    return (results, results)

def reset_textbox():
    return gr.Textbox.update(value='')

with gr.Blocks(css = """#col_container {width: 95%; margin-left: auto; margin-right: auto;}
                #chatbot {height: 500px; overflow: auto;}
                .chat_wrap_space {margin-left: 0.5em} """) as demo:

    state_chatbot = gr.State([])

    with gr.Column(elem_id='col_container'):
        gr.Markdown(f"## {TITLE}\n\n\n{ABSTRACT}")

        with gr.Accordion("Context Setting", open=False):
            context_txtbox = gr.Textbox(placeholder="Surrounding information to AI", label="Enter Context")
            hidden_txtbox = gr.Textbox(placeholder="", label="Order", visible=False)

        chatbot = gr.Chatbot(elem_id='chatbot', label="Alpaca-LoRA")
        instruction_txtbox = gr.Textbox(placeholder="What do you want to say to AI?", label="Instruction")
        send_prompt_btn = gr.Button(value="Send Prompt")

        gr.Examples(
            examples=[
                ["1️⃣", "List all Canadian provinces in alphabetical order."],
                ["1️⃣ ▶️ 1️⃣", "Which ones are on the east side?"],
                ["1️⃣ ▶️ 2️⃣", "What foods are famous in each province?"],
                ["1️⃣ ▶️ 3️⃣", "What about sightseeing? or landmarks?"],
                ["2️⃣", "Tell me about alpacas."],
                ["2️⃣ ▶️ 1️⃣", "What other animals are living in the same area?"],
                ["2️⃣ ▶️ 2️⃣", "Are they the same species?"],
                ["2️⃣ ▶️ 3️⃣", "Write a Python program to return those species"],
                ["3️⃣", "Tell me about the king of France in 2019."],                
                ["4️⃣", "Write a Python program that prints the first 10 Fibonacci numbers."],                
            ], 
            inputs=[
                hidden_txtbox, instruction_txtbox
            ],
            label="Examples. ▶️ symbol indicates follow-up prompts"
        )

        gr.Markdown(f"{BOTTOM_LINE}")

    send_prompt_btn.click(
        chat, 
        [context_txtbox, instruction_txtbox, state_chatbot],
        [state_chatbot, chatbot],
        batch=True,
        max_batch_size=4
    )
    send_prompt_btn.click(
        reset_textbox, 
        [], 
        [instruction_txtbox],
    )

demo.queue().launch(server_port=6006)
