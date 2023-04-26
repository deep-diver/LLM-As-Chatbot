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

            with gr.Accordion("Constrol Panel", open=False, visible=not args.chat_only_mode):
                with gr.Column():
                    with gr.Column():
                        gr.Markdown("#### GenConfig for **response** text generation")
                        with gr.Row():
                            res_temp = gr.Slider(0.0, 2.0, global_vars.gen_config.temperature, step=0.1, label="temp", interactive=True)
                            res_topp = gr.Slider(0.0, 2.0, global_vars.gen_config.top_p, step=0.1, label="top_p", interactive=True)
                            res_topk = gr.Slider(20, 1000, global_vars.gen_config.top_k, step=1, label="top_k", interactive=True)
                            res_rpen = gr.Slider(0.0, 2.0, global_vars.gen_config.repetition_penalty, step=0.1, label="rep_penalty", interactive=True)
                            res_mnts = gr.Slider(64, 1024, global_vars.gen_config.max_new_tokens, step=1, label="max_new_tokens", interactive=True)                            
                            res_beams = gr.Slider(1, 4, global_vars.gen_config.num_beams, step=1, label="num_beams")
                            res_cache = gr.Radio([True, False], value=global_vars.gen_config.use_cache, label="use_cache", interactive=True)
                            res_sample = gr.Radio([True, False], value=global_vars.gen_config.do_sample, label="do_sample", interactive=True)
                            res_eosid = gr.Number(value=global_vars.gen_config.eos_token_id, visible=False, precision=0)
                            res_padid = gr.Number(value=global_vars.gen_config.pad_token_id, visible=False, precision=0)

                    with gr.Column():
                        gr.Markdown("#### GenConfig for **summary** text generation")
                        with gr.Row():
                            sum_temp = gr.Slider(0.0, 2.0, global_vars.gen_config_summarization.temperature, step=0.1, label="temperature", interactive=True)
                            sum_topp = gr.Slider(0.0, 2.0, global_vars.gen_config_summarization.top_p, step=0.1, label="top_p", interactive=True)
                            sum_topk = gr.Slider(20, 1000, global_vars.gen_config_summarization.top_k, step=1, label="top_k", interactive=True)
                            sum_rpen = gr.Slider(0.0, 2.0, global_vars.gen_config_summarization.repetition_penalty, step=0.1, label="rep_penalty", interactive=True)
                            sum_mnts = gr.Slider(64, 1024, global_vars.gen_config_summarization.max_new_tokens, step=1, label="max_new_tokens", interactive=True)
                            sum_beams = gr.Slider(1, 8, global_vars.gen_config_summarization.num_beams, step=1, label="num_beams", interactive=True)
                            sum_cache = gr.Radio([True, False], value=global_vars.gen_config_summarization.use_cache, label="use_cache", interactive=True)
                            sum_sample = gr.Radio([True, False], value=global_vars.gen_config_summarization.do_sample, label="do_sample", interactive=True)
                            sum_eosid = gr.Number(value=global_vars.gen_config_summarization.eos_token_id, visible=False, precision=0)
                            sum_padid = gr.Number(value=global_vars.gen_config_summarization.pad_token_id, visible=False, precision=0)

                    with gr.Column():
                        gr.Markdown("#### Context managements")
                        with gr.Row():
                            ctx_num_lconv = gr.Slider(2, 6, 3, step=1, label="num of last talks to keep", interactive=True)
                            ctx_sum_prompt = gr.Textbox(
                                "summarize our conversations. what have we discussed about so far?",
                                label="design a prompt to summarize the conversations"
                            )

            with gr.Accordion("Acknowledgements", open=False, visible=not args.chat_only_mode):
                gr.Markdown(f"{BOTTOM_LINE}")

        view_selector.change(
            toggle_inspector,
            view_selector,
            context_inspector_section
        )
                
        send_event = instruction_txtbox.submit(
            chat_interface,
            [instruction_txtbox, chat_state,
            ctx_num_lconv, ctx_sum_prompt,
            res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
            sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid],
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
