from strings import TITLE, ABSTRACT, BOTTOM_LINE

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
    if ctx.strip() == "":
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
    contexts,
    instructions, 
    state_chatbots
):
    results = []

    instruct_prompts = [generate_prompt(instruct, ctx) for ctx, instruct in zip(contexts, instructions)]

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
                #chatbot {height: 500px; overflow: auto;}""") as demo:

    state_chatbot = gr.State([])

    with gr.Column(elem_id='col_container'):
        gr.Markdown(f"## {TITLE}\n\n\n{ABSTRACT}")

        context_txtbox = gr.Textbox(placeholder="Explain surrounding information to AI", label="Enter Context")
        chatbot = gr.Chatbot(elem_id='chatbot', label="Alpaca-LoRA")
        instruction_txtbox = gr.Textbox(placeholder="What do you want to say to AI?", label="Enter Instruction")
        send_prompt_btn = gr.Button(value="Send Prompt")

        gr.Examples(
            examples=[
                ["Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.", "List all Canadian provinces in alphabetical order."],
                ["Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.", "Tell me about the king of France in 2019."],
                ["Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.", "Write a Python program that prints the first 10 Fibonacci numbers."],
                ["Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.", "Tell me about alpacas."]
            ], 
            inputs=[
                context_txtbox, 
                instruction_txtbox
            ]
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
