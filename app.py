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
from utils import generate_prompt, post_processes, post_process

def chat(
    contexts,
    instructions, 
    state_chatbots
):
    print("-------state_chatbots------")
    print(state_chatbots)
    results = []

    instruct_prompts = [
        generate_prompt(instruct, histories, ctx) 
        for ctx, instruct, histories in zip(contexts, instructions, state_chatbots)
    ]
        
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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Gradio Application for Alpaca-LoRA as a chatbot service"
    )
    # Dataset related.
    parser.add_argument(
        "--base_url",
        help="huggingface hub url",
        default="decapoda-research/llama-7b-hf",
        type=str,
    )
    parser.add_argument(
        "--ft_ckpt_url",
        help="huggingface hub url",
        default="tloen/alpaca-lora-7b",
        type=str,
    )
    parser.add_argument(
        "--port",
        help="port to serve app",
        default=6006,
        type=int,
    )
    parser.add_argument(
        "--api_open",
        help="do you want to open as API",
        default="no",
        type=str,
    )

    return parser.parse_args()

def run(args):
    global model, tokenizer

    model, tokenizer = load_model(
        base=args.base_url,
        finetuned=args.ft_ckpt_url
    )

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
            max_batch_size=4,
            api_name="text_gen"
        )
        send_prompt_btn.click(
            reset_textbox, 
            [], 
            [instruction_txtbox],
        )

    demo.queue(
        api_open=False if args.api_open == "no" else True
    ).launch(
        server_port=args.port
    )

if __name__ == "__main__":
    args = parse_args()
    run(args)
