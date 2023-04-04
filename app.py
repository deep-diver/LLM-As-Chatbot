from strings import TITLE, ABSTRACT, BOTTOM_LINE
from strings import DEFAULT_EXAMPLES
from strings import SPECIAL_STRS
from styles import PARENT_BLOCK_CSS

from constants import num_of_characters_to_keep

import time
import gradio as gr

from args import parse_args
from model import load_model
from gen import get_output_batch, StreamModel
from utils import generate_prompt, post_processes_batch, post_process_stream, get_generation_config, common_post_process

def chat_stream(
    context,
    instruction,
    state_chatbot,
):
    if len(context) > 500 or len(instruction) > 150:
        raise gr.Error("context or prompt is too long!")
    
    bot_summarized_response = ''
    # user input should be appropriately formatted (don't be confused by the function name)
    instruction_display = common_post_process(instruction)
    instruction_prompt, conv_length = generate_prompt(instruction, state_chatbot, context)
    
    if conv_length > num_of_characters_to_keep:
        instruction_prompt = generate_prompt(SPECIAL_STRS["summarize"], state_chatbot, context)[0]
        
        state_chatbot = state_chatbot + [
            (
                None, 
                "![](https://s2.gifyu.com/images/icons8-loading-circle.gif) too long conversations, so let's summarize..."
            )
        ]
        yield (state_chatbot, state_chatbot, context)
        
        bot_summarized_response = get_output_batch(
            model, tokenizer, [instruction_prompt], gen_config_summarization
        )[0]
        bot_summarized_response = bot_summarized_response.split("### Response:")[-1].strip()
        
        state_chatbot[-1] = (
            None, 
            "✅ summarization is done and set as context"
        )
        print(f"bot_summarized_response: {bot_summarized_response}")
        yield (state_chatbot, state_chatbot, f"{context}. {bot_summarized_response}")
        
    instruction_prompt = generate_prompt(instruction, state_chatbot, f"{context} {bot_summarized_response}")[0]
    
    bot_response = stream_model(
        instruction_prompt,
        max_tokens=256,
        temperature=1,
        top_p=0.9
    )
    
    instruction_display = None if instruction_display == SPECIAL_STRS["continue"] else instruction_display
    state_chatbot = state_chatbot + [(instruction_display, None)]
    
    prev_index = 0
    agg_tokens = ""
    cutoff_idx = 0
    for tokens in bot_response:
        tokens = tokens.strip()
        cur_token = tokens[prev_index:]
        
        if "#" in cur_token and agg_tokens == "":
            cutoff_idx = tokens.find("#")
            agg_tokens = tokens[cutoff_idx:]

        if agg_tokens != "":
            if len(agg_tokens) < len("### Instruction:") :
                agg_tokens = agg_tokens + cur_token
            elif len(agg_tokens) >= len("### Instruction:"):
                if tokens.find("### Instruction:") > -1:
                    processed_response, _ = post_process_stream(tokens[:tokens.find("### Instruction:")].strip())

                    state_chatbot[-1] = (
                        instruction_display, 
                        processed_response
                    )
                    yield (state_chatbot, state_chatbot, f"{context} {bot_summarized_response}")
                    break
                else:
                    agg_tokens = ""
                    cutoff_idx = 0

        if agg_tokens == "":
            processed_response, to_exit = post_process_stream(tokens)
            state_chatbot[-1] = (instruction_display, processed_response)
            yield (state_chatbot, state_chatbot, f"{context} {bot_summarized_response}")

            if to_exit:
                break

        prev_index = len(tokens)

    yield (
        state_chatbot,
        state_chatbot,
        f"{context} {bot_summarized_response}"
    )

def chat_batch(
    contexts,
    instructions, 
    state_chatbots,
):
    state_results = []
    ctx_results = []

    instruct_prompts = [
        generate_prompt(instruct, histories, ctx)[0]
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
    global model, stream_model, tokenizer, generation_config, gen_config_summarization, batch_enabled
    
    batch_enabled = True if args.batch_size > 1 else False    

    model, tokenizer = load_model(
        base=args.base_url,
        finetuned=args.ft_ckpt_url,
        multi_gpu=args.multi_gpu
    )    
    
    generation_config = get_generation_config(
        args.gen_config_path
    )
    gen_config_summarization = get_generation_config(
        "gen_config_summarization.yaml"
    )
    
    if not batch_enabled:
        stream_model = StreamModel(model, tokenizer)
    
    with gr.Blocks(css=PARENT_BLOCK_CSS) as demo:
        state_chatbot = gr.State([])

        with gr.Column(elem_id='col_container'):
            gr.Markdown(f"## {TITLE}\n\n\n{ABSTRACT}")

            with gr.Accordion("Context Setting", open=False):
                context_txtbox = gr.Textbox(placeholder="Surrounding information to AI", label="Context")
                hidden_txtbox = gr.Textbox(placeholder="", label="Order", visible=False)

            chatbot = gr.Chatbot(elem_id='chatbot', label="Alpaca-LoRA")
            instruction_txtbox = gr.Textbox(placeholder="What do you want to say to AI?", label="Instruction")
            with gr.Row():
                cancel_btn = gr.Button(value="Cancel")
                reset_btn = gr.Button(value="Reset")
            
            with gr.Accordion("Helper Buttons", open=False):
                gr.Markdown(f"`Continue` lets AI to complete the previous incomplete answers. `Summarize` lets AI to summarize the conversations so far.")
                continue_txtbox = gr.Textbox(value=SPECIAL_STRS["continue"], visible=False)
                summrize_txtbox = gr.Textbox(value=SPECIAL_STRS["summarize"], visible=False)
                
                continue_btn = gr.Button(value="Continue")
                summarize_btn = gr.Button(value="Summarize")

            gr.Markdown("#### Examples")
            for _, (category, examples) in enumerate(DEFAULT_EXAMPLES.items()):
                with gr.Accordion(category, open=False):
                    if category == "Identity":
                        for item in examples:
                            with gr.Accordion(item["title"], open=False):
                                gr.Examples(
                                    examples=item["examples"],
                                    inputs=[
                                        hidden_txtbox, context_txtbox, instruction_txtbox
                                    ],
                                    label=None
                                )
                    else:
                        for item in examples:
                            with gr.Accordion(item["title"], open=False):
                                gr.Examples(
                                    examples=item["examples"],
                                    inputs=[
                                        hidden_txtbox, instruction_txtbox
                                    ],
                                    label=None
                                )

            gr.Markdown(f"{BOTTOM_LINE}")

        send_event = instruction_txtbox.submit(
            chat_batch if batch_enabled else chat_stream, 
            [context_txtbox, instruction_txtbox, state_chatbot],
            [state_chatbot, chatbot, context_txtbox],
            batch=batch_enabled,
            max_batch_size=args.batch_size,
        )
        reset_event = instruction_txtbox.submit(
            reset_textbox, 
            [], 
            [instruction_txtbox],
        )
        
        continue_event = continue_btn.click(
            chat_batch if batch_enabled else chat_stream, 
            [context_txtbox, continue_txtbox, state_chatbot],
            [state_chatbot, chatbot, context_txtbox],
            batch=batch_enabled,
            max_batch_size=args.batch_size,
        )
        reset_continue_event = continue_btn.click(
            reset_textbox, 
            [], 
            [instruction_txtbox],
        )
        
        summarize_event = summarize_btn.click(
            chat_batch if batch_enabled else chat_stream, 
            [context_txtbox, summrize_txtbox, state_chatbot],
            [state_chatbot, chatbot, context_txtbox],
            batch=batch_enabled,
            max_batch_size=args.batch_size,
        )
        summarize_reset_event = summarize_btn.click(
            reset_textbox, 
            [], 
            [instruction_txtbox],
        )
        
        cancel_btn.click(
            None, None, None, 
            cancels=[
                send_event, continue_event, summarize_event
            ]
        )

        reset_btn.click(
            reset_everything,
            [context_txtbox, instruction_txtbox, state_chatbot],
            [state_chatbot, chatbot, context_txtbox, instruction_txtbox],
            cancels=[
                send_event, continue_event, summarize_event
            ]            
        )        

    demo.queue(
        concurrency_count=2,
        max_size=100,
        api_open=args.api_open
    ).launch(
        max_threads=2,
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0",
    )

if __name__ == "__main__":
    args = parse_args()
    run(args)
