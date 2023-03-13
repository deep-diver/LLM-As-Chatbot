from typing import Tuple
import os
import sys
import argparse
import torch
import time
import json
import asyncio
from functools import partial
import gradio as gr

from pathlib import Path
from typing import List

import torch.distributed as dist
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from strings import TITLE, ABSTRACT, EXAMPLES

history = []

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    dist.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

def init_generator(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    return generator

generator = init_generator(
    "weights/13B",
    "weights/tokenizer/tokenizer.model",
    256,
    1,
)

if dist.get_rank() == 0:
    def get_output(
        prompt: str,
        max_gen_len: int = 256,
        temperature: float = 0.8, 
        top_p: float = 0.95):

        print(prompt)
        prompts = [prompt]
        
        dist.broadcast_object_list([prompts, max_gen_len, temperature, top_p])
        
        results = generator.generate(
            prompts, 
            max_gen_len=max_gen_len, 
            temperature=temperature, 
            top_p=top_p
        )

        return results

    def chat(
        user_input, 
        include_input,
        truncate,    
        top_p, 
        temperature, 
        max_gen_len, 
        state_chatbot
    ):
        bot_response = get_output(
            prompt=user_input,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p)[0]

        # remove the first phrase identical to user prompt
        if not include_input:
            bot_response = bot_response[len(user_input):]
        bot_response = bot_response.replace("\n", "<br>")

        # trip the last phrase
        if truncate:
            try:
                bot_response = bot_response[:bot_response.rfind(".")+1]
            except:
                pass

        history.append({
            "role": "user",
            "content": user_input
        })
        history.append({
            "role": "system",
            "content": bot_response
        })    

        state_chatbot = state_chatbot + [(user_input, None)]

        prev_len = 0
        response = ""
        for word in bot_response.split(" "):
            time.sleep(0.1)
            response += word + " "
            current_pair = (user_input, response)
            state_chatbot[-1] = current_pair
            yield state_chatbot, state_chatbot

    def reset_textbox():
        return gr.update(value='')    
    
    with gr.Blocks(css = """#col_container {width: 95%; margin-left: auto; margin-right: auto;}
                    #chatbot {height: 400px; overflow: auto;}""") as demo:

        state_chatbot = gr.State([])

        with gr.Column(elem_id='col_container'):
            gr.Markdown(f"## {TITLE}\n\n\n\n{ABSTRACT}")

            with gr.Accordion("Example prompts", open=False):
                example_str = "\n"
                for example in EXAMPLES:
                    example_str += f"- {example}\n"

                gr.Markdown(example_str)        

            chatbot = gr.Chatbot(elem_id='chatbot')
            textbox = gr.Textbox(placeholder="Enter a prompt")

            with gr.Accordion("Parameters", open=False):
                include_input = gr.Checkbox(value=True, label="Do you want to include the input in the generated text?")
                truncate = gr.Checkbox(value=True, label="Truncate the unfinished last words?")

                max_gen_len = gr.Slider(minimum=20, maximum=512, value=256, step=1, interactive=True, label="Max Genenration Length",)
                top_p = gr.Slider(minimum=-0, maximum=1.0, value=1.0, step=0.05, interactive=True, label="Top-p (nucleus sampling)",)
                temperature = gr.Slider(minimum=-0, maximum=5.0, value=1.0, step=0.1, interactive=True, label="Temperature",)

        textbox.submit(
            chat, 
            [textbox, include_input, truncate, top_p, temperature, max_gen_len, state_chatbot],
            [state_chatbot, chatbot]
        )
        textbox.submit(reset_textbox, [], [textbox])    
    
    demo.queue().launch(share=True, server_port=6006)
else:
    while True:
        time.sleep(0.5)
        config = [None] * 4
        try:
            dist.broadcast_object_list(config)
            generator.generate(
                config[0], max_gen_len=config[1], temperature=config[2], top_p=config[3]
            )
        except:
            pass
