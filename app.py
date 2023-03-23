from strings import TITLE, ABSTRACT, BOTTOM_LINE
from strings import DEFAULT_EXAMPLES
from strings import SPECIAL_STRS
from styles import PARENT_BLOCK_CSS

import gradio as gr

from args import parse_args
from model import load_model
from gen import get_output_batch, StreamModel
from utils import generate_prompt, post_processes_batch, post_process_stream, get_generation_config

import asyncio

async def chat_stream(
    context,
    instruction,
    state_chatbot,
):
    instruction_prompt = generate_prompt(instruction, state_chatbot, context)    
    bot_response = model(
        instruction_prompt,
        max_tokens=128,
        temperature=0.90,
        top_p=0.75
    )
    
    instruction = '' if instruction == SPECIAL_STRS["continue"] else instruction
    state_chatbot = state_chatbot + [(instruction, None)]
    
    for tokens in bot_response:
        asyncio.sleep(0.1)
        tokens, to_stop = post_process_stream(tokens.strip())
        state_chatbot[-1] = (instruction, tokens)
        yield (state_chatbot, state_chatbot, context)
        
        if to_stop:
            break

    yield (
        state_chatbot,
        state_chatbot,
        gr.Textbox.update(value=tokens) if instruction == SPECIAL_STRS["summarize"] else context
    )

def chat_batch(
    contexts,
    instructions, 
    state_chatbots,
):
    state_results = []
    ctx_results = []

    instruct_prompts = [
        generate_prompt(instruct, histories, ctx) 
        for ctx, instruct, histories in zip(contexts, instructions, state_chatbots)
    ]
        
    bot_responses = get_output_batch(
        model, tokenizer, instruct_prompts, generation_config
    )
    bot_responses = post_processes_batch(bot_responses)

    for ctx, instruction, bot_response, state_chatbot in zip(contexts, instructions, bot_responses, state_chatbots):
        new_state_chatbot = state_chatbot + [('' if instruction == SPECIAL_STRS["continue"] else instruction, bot_response)]
        ctx_results.append(gr.Textbox.update(value=bot_response) if instruction == SPECIAL_STRS["summarize"] else ctx)
        state_results.append(new_state_chatbot)

    return (state_results, state_results, ctx_results)

def reset_textbox():
    return gr.Textbox.update(value='')

def run(args):
    global model, tokenizer, generation_config, batch_enabled
    
    batch_enabled = True if args.batch_size > 1 else False    

    model, tokenizer = load_model(
        base=args.base_url,
        finetuned=args.ft_ckpt_url
    )    
    
    generation_config = get_generation_config(
        args.gen_config_path
    )
    
    if not batch_enabled:
        model = StreamModel(model, tokenizer)
        # model.generation_config = generation_config
    
    with gr.Blocks(css=PARENT_BLOCK_CSS) as demo:
        state_chatbot = gr.State([])

        with gr.Column(elem_id='col_container'):
            gr.Markdown(f"## {TITLE}\n\n\n{ABSTRACT}")

            with gr.Accordion("Context Setting", open=False):
                context_txtbox = gr.Textbox(placeholder="Surrounding information to AI", label="Enter Context")
                hidden_txtbox = gr.Textbox(placeholder="", label="Order", visible=False)

            chatbot = gr.Chatbot(elem_id='chatbot', label="Alpaca-LoRA")
            instruction_txtbox = gr.Textbox(placeholder="What do you want to say to AI?", label="Instruction")
            send_prompt_btn = gr.Button(value="Send Prompt")
            
            with gr.Accordion("Helper Buttons", open=False):
                gr.Markdown(f"`Continue` lets AI to complete the previous incomplete answers. `Summarize` lets AI to summarize the conversations so far.")
                continue_txtbox = gr.Textbox(value=SPECIAL_STRS["continue"], visible=False)
                summrize_txtbox = gr.Textbox(value=SPECIAL_STRS["summarize"], visible=False)
                
                continue_btn = gr.Button(value="Continue")
                summarize_btn = gr.Button(value="Summarize")

            gr.Markdown("#### Examples")
            for idx, examples in enumerate(DEFAULT_EXAMPLES):
                with gr.Accordion(examples["title"], open=False):
                    gr.Examples(
                        examples=examples["examples"], 
                        inputs=[
                            hidden_txtbox, instruction_txtbox
                        ],
                        label=None
                    )

            gr.Markdown(f"{BOTTOM_LINE}")
            
        send_prompt_btn.click(
            chat_batch if batch_enabled else chat_stream, 
            [context_txtbox, instruction_txtbox, state_chatbot],
            [state_chatbot, chatbot, context_txtbox],
            batch=batch_enabled,
            max_batch_size=args.batch_size,
        )
        send_prompt_btn.click(
            reset_textbox, 
            [], 
            [instruction_txtbox],
        )
        
        continue_btn.click(
            chat_batch if batch_enabled else chat_stream, 
            [context_txtbox, continue_txtbox, state_chatbot],
            [state_chatbot, chatbot, context_txtbox],
            batch=batch_enabled,
            max_batch_size=args.batch_size,
        )
        continue_btn.click(
            reset_textbox, 
            [], 
            [instruction_txtbox],
        )
        
        summarize_btn.click(
            chat_batch if batch_enabled else chat_stream, 
            [context_txtbox, summrize_txtbox, state_chatbot],
            [state_chatbot, chatbot, context_txtbox],
            batch=batch_enabled,
            max_batch_size=args.batch_size,
        )
        summarize_btn.click(
            reset_textbox, 
            [], 
            [instruction_txtbox],
        )              

    demo.queue(
        concurrency_count=2,
        max_size=100,
        api_open=False if args.api_open == "no" else True
    ).launch(
        max_threads=2,
        share=False if args.share == "no" else True,
        server_port=args.port
    )

if __name__ == "__main__":
    args = parse_args()
    run(args)
