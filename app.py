import time
import gradio as gr

import global_vars
import alpaca

from args import parse_args
from miscs.strings import TITLE, ABSTRACT, BOTTOM_LINE
from miscs.strings import DEFAULT_EXAMPLES
from miscs.styles import PARENT_BLOCK_CSS
from miscs.strings import SPECIAL_STRS

from utils import get_chat_interface

def reset_textbox():
    return gr.Textbox.update(value='')

def reset_everything(
    context_txtbox, 
    instruction_txtbox, 
    state_chatbot):

    state_chatbot = []
    
    return (
        state_chatbot,
        state_chatbot,
        gr.Textbox.update(value=''),
        gr.Textbox.update(value=''),
    )

def run(args):
    global_vars.initialize_globals(args)
    batch_enabled = global_vars.batch_enabled
    chat_interface = get_chat_interface(global_vars.model_type, batch_enabled)
    
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
            chat_interface,
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
            chat_interface,
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
            chat_interface,
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
