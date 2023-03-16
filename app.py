from strings import TITLE, ABSTRACT

import os
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
    if ctx is None:
        ctx = "This is the initial conversation"

    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Input:
{ctx}

### Response:"""

def post_process(bot_response):
    bot_response = bot_response.split("### Response:")[1].strip()
    return bot_response.replace("\n", "<br>").replace(" ", "&nbsp;")    

def post_processes(bot_responses):
    return [post_process(r) for r in bot_responses]

def chat(
    prompts,
    state_chatbots
):
    print(prompts)
    print(state_chatbots)
    results = []

    instruct_prompts = [generate_prompt(prompt, None) for prompt in prompts]

    bot_responses = get_output(
        model, tokenizer, instruct_prompts, None
    )
    print(bot_responses)
    bot_responses = post_processes(bot_responses)

    print("zipping...")
    sub_results = []
    for prompt, bot_response, state_chatbot in zip(prompts, bot_responses, state_chatbots):
        print(prompt)
        print(bot_response)
        print(state_chatbot)
        new_state_chatbot = state_chatbot + [(prompt, bot_response)]
        print(new_state_chatbot)
        results.append(new_state_chatbot)

    print(results)
    print(len(results))

    return (results, results)

def reset_textbox():
    return gr.Textbox.update(value='')

with gr.Blocks(css = """#col_container {width: 95%; margin-left: auto; margin-right: auto;}
                #chatbot {height: 600px; overflow: auto;}""") as demo:

    state_chatbot = gr.State([])

    with gr.Column(elem_id='col_container'):
        gr.Markdown(f"## {TITLE}\n\n\n{ABSTRACT}")

        chatbot = gr.Chatbot(elem_id='chatbot', label="Alpaca-LoRA")
        textbox = gr.Textbox(placeholder="Enter a prompt")

    textbox.submit(
        chat, 
        [textbox, state_chatbot],
        [state_chatbot, chatbot],
        batch=True,
    )
    textbox.submit(
        reset_textbox, 
        [], 
        [textbox],
    )

demo.queue().launch(server_port=6006)