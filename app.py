import time
import gradio as gr
import global_vars
from args import parse_args
from utils import get_chat_interface, get_chat_manager

from miscs.styles import PARENT_BLOCK_CSS
from miscs.strings import BOTTOM_LINE

def reset_textbox():
    return gr.Textbox.update(value='')

# [chatbot, chat_state, inspector, instruction_txtbox],
def reset_everything():
    return (
        [],
        {"ppmanager": get_chat_manager(global_vars.model_type)},
        "",
        "",
    )

def toggle_inspector(view_selector):
    if view_selector == "with context inspector":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)        

def run(args):
    global_vars.initialize_globals(args)
    chat_interface = get_chat_interface(global_vars.model_type)

    with gr.Blocks(css=PARENT_BLOCK_CSS, theme='ParityError/Anime') as demo:
        chat_state = gr.State({
            "ppmanager": get_chat_manager(global_vars.model_type)
        })

        with gr.Column(elem_id='col-container'):
            view_selector = gr.Radio(
                ["with context inspector", "without context inspector"],
                value="without context inspector",
                label="View Selector", 
                info="How do you like to use this application?",
                visible=not args.chat_only_mode
            )
            
            with gr.Row():
                with gr.Column(elem_id='chatbot-wrapper'):
                    chatbot = gr.Chatbot(elem_id='chatbot', label=global_vars.model_type)
                    instruction_txtbox = gr.Textbox(placeholder="What do you want to say to AI?", label="Instruction")
            
                    with gr.Row():
                        cancel_btn = gr.Button(value="Cancel")
                        reset_btn = gr.Button(value="Reset")
                        
                with gr.Column(visible=False) as context_inspector_section:
                    gr.Markdown("#### What model actually sees")
                    inspector = gr.Textbox(label="", lines=28, max_lines=28, interactive=False)

            with gr.Accordion("Acknowledgements", open=False, visible=not args.chat_only_mode):
                gr.Markdown(f"{BOTTOM_LINE}")

        view_selector.change(
            toggle_inspector,
            view_selector,
            context_inspector_section
        )
                
        send_event = instruction_txtbox.submit(
            chat_interface,
            [instruction_txtbox, chat_state],
            [instruction_txtbox, chatbot, inspector, chat_state],
        )
        
        cancel_btn.click(
            None, None, None, 
            cancels=[send_event]
        )

        reset_btn.click(
            reset_everything,
            None,
            [chatbot, chat_state, inspector, instruction_txtbox],
            cancels=[send_event]            
        )

    demo.queue(
        concurrency_count=2,
        max_size=100,
    ).launch(
        max_threads=10,
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0",
    )

if __name__ == "__main__":
    args = parse_args()
    run(args)
