import time
import json
from os import listdir
from os.path import isfile, join
import argparse
import gradio as gr
import extra_args
import global_vars
from chats import central
from transformers import AutoModelForCausalLM
from miscs.styles import MODEL_SELECTION_CSS
from miscs.js import GET_LOCAL_STORAGE, UPDATE_LEFT_BTNS_STATE
from utils import get_chat_interface, get_chat_manager

ex_file = open("examples.txt", "r")
examples = ex_file.read().split("\n")
ex_btns = []

chl_file = open("channels.txt", "r")
channels = chl_file.read().split("\n")
channel_btns = []

response_configs = [
    f"configs/response_configs/{f}"
    for f in listdir("configs/response_configs")
    if isfile(join("configs/response_configs", f))
]

summarization_configs = [
    f"configs/summarization_configs/{f}"
    for f in listdir("configs/summarization_configs")
    if isfile(join("configs/summarization_configs", f))
]

model_info = json.load(open("model_cards.json"))

def channel_num(btn_title):
    choice = 0

    for idx, channel in enumerate(channels):
        if channel == btn_title:
            choice = idx

    return choice


def set_chatbot(btn, ld, state):
    choice = channel_num(btn)

    res = [state["ppmanager_type"].from_json(json.dumps(ppm_str)) for ppm_str in ld]
    empty = len(res[choice].pingpongs) == 0
    return (res[choice].build_uis(), choice, gr.update(visible=empty), gr.update(interactive=not empty))


def set_example(btn):
    return btn, gr.update(visible=False)


def set_popup_visibility(ld, example_block):
    return example_block


def move_to_second_view(btn):
    info = model_info[btn]

    return (
        gr.update(visible=False),
        gr.update(visible=True),
        info["thumb"],
        f"**Model name**\n: {btn}",
        f"**Parameters**\n: {info['parameters']}",
        f"**🤗 Hub(base)**\n: {info['hub(base)']}",
        f"**🤗 Hub(ckpt)**\n: {info['hub(ckpt)']}",
        "",
    )


def move_to_first_view():
    return (gr.update(visible=True), gr.update(visible=False))


def download_completed(
    model_name,
    model_base,
    model_ckpt,
    gen_config_path,
    gen_config_sum_path,
    multi_gpu,
    force_download,
):

    tmp_args = extra_args.parse_args()
    tmp_args.base_url = model_base.split(":")[-1].split("</p")[0].strip()
    tmp_args.ft_ckpt_url = model_ckpt.split(":")[-1].split("</p")[0].strip()
    tmp_args.gen_config_path = gen_config_path
    tmp_args.gen_config_summarization_path = gen_config_sum_path
    tmp_args.multi_gpu = multi_gpu
    tmp_args.force_download_ckpt = force_download

    global_vars.initialize_globals(tmp_args)
    return "Download completed!"

def move_to_third_view():
    gen_config = global_vars.gen_config
    gen_sum_config = global_vars.gen_config_summarization

    return (
        "Preparation done!",
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(label=global_vars.model_type),
        {
            "ppmanager_type": get_chat_manager(global_vars.model_type),
            "model_type": global_vars.model_type,
        },
        gen_config.temperature,
        gen_config.top_p,
        gen_config.top_k,
        gen_config.repetition_penalty,
        gen_config.max_new_tokens,
        gen_config.num_beams,
        gen_config.use_cache,
        gen_config.do_sample,
        gen_config.eos_token_id,
        gen_config.pad_token_id,
        gen_sum_config.temperature,
        gen_sum_config.top_p,
        gen_sum_config.top_k,
        gen_sum_config.repetition_penalty,
        gen_sum_config.max_new_tokens,
        gen_sum_config.num_beams,
        gen_sum_config.use_cache,
        gen_sum_config.do_sample,
        gen_sum_config.eos_token_id,
        gen_sum_config.pad_token_id,
    )


def toggle_inspector(view_selector):
    if view_selector == "with context inspector":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def reset_chat(idx, ld, state):
    res = [state["ppmanager_type"].from_json(json.dumps(ppm_str)) for ppm_str in ld]
    res[idx].pingpongs = []
        
    return (
        "",
        [],
        str(res),
        gr.update(visible=True),
        gr.update(interactive=False),
    )

def rollback_last(idx, ld, state):
    res = [state["ppmanager_type"].from_json(json.dumps(ppm_str)) for ppm_str in ld]
    last_user_message = res[idx].pingpongs[-1].ping
    res[idx].pingpongs = res[idx].pingpongs[:-1]
    
    return (
        last_user_message,
        res[idx].build_uis(),
        str(res),
        gr.update(interactive=False)
    )

def main(root_path):
    with gr.Blocks(css=MODEL_SELECTION_CSS, theme='gradio/soft') as demo:
        with gr.Column() as model_choice_view:
            gr.Markdown("# Choose a Model", elem_classes=["center"])
            with gr.Row(elem_id="container"):
                with gr.Column():
                    gr.Markdown("## < 10B")
                    with gr.Row(elem_classes=["sub-container"]):
                        with gr.Column(min_width=20):
                            t5_vicuna_3b = gr.Button(
                                "t5-vicuna-3b",
                                elem_id="t5-vicuna-3b",
                                elem_classes=["square"],
                            )
                            gr.Markdown("T5 Vicuna", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            flan3b = gr.Button(
                                "flan-3b", elem_id="flan-3b", elem_classes=["square"]
                            )
                            gr.Markdown("Flan-XL", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            camel5b = gr.Button(
                                "camel-5b", elem_id="camel-5b", elem_classes=["square"]
                            )
                            gr.Markdown("Camel", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            alpaca_lora7b = gr.Button(
                                "alpaca-lora-7b",
                                elem_id="alpaca-lora-7b",
                                elem_classes=["square"],
                            )
                            gr.Markdown("Alpaca-LoRA", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            stablelm7b = gr.Button(
                                "stablelm-7b",
                                elem_id="stablelm-7b",
                                elem_classes=["square"],
                            )
                            gr.Markdown("StableLM", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            os_stablelm7b = gr.Button(
                                "os-stablelm-7b",
                                elem_id="os-stablelm-7b",
                                elem_classes=["square"],
                            )
                            gr.Markdown("OS+StableLM", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            gpt4_alpaca_7b = gr.Button(
                                "gpt4-alpaca-7b",
                                elem_id="gpt4-alpaca-7b",
                                elem_classes=["square"],
                            )
                            gr.Markdown("GPT4-Alpaca-LoRA", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            mpt_7b = gr.Button(
                                "mpt-7b", elem_id="mpt-7b", elem_classes=["square"]
                            )
                            gr.Markdown("MPT", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            redpajama_7b = gr.Button(
                                "redpajama-7b",
                                elem_id="redpajama-7b",
                                elem_classes=["square"],
                            )
                            gr.Markdown("RedPajama(7B)", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            vicuna_7b = gr.Button(
                                "vicuna-7b", elem_id="vicuna-7b", elem_classes=["square"]
                            )
                            gr.Markdown("Vicuna", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            llama_deus_7b = gr.Button(
                                "llama-deus-7b",
                                elem_id="llama-deus-7b",
                                elem_classes=["square"],
                            )
                            gr.Markdown("LLaMA Deus", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            evolinstruct_vicuna_7b = gr.Button(
                                "evolinstruct-vicuna-7b",
                                elem_id="evolinstruct-vicuna-7b",
                                elem_classes=["square"],
                            )
                            gr.Markdown("EvolInstruct Vicuna", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            alpacoom_7b = gr.Button(
                                "alpacoom-7b",
                                elem_id="alpacoom-7b",
                                elem_classes=["square"],
                            )
                            gr.Markdown("Alpacoom", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            baize_7b = gr.Button(
                                "baize-7b",
                                elem_id="baize-7b",
                                elem_classes=["square"],
                            )
                            gr.Markdown("Baize", elem_classes=["center"])                        
                            
                        with gr.Column(min_width=20):
                            guanaco_7b = gr.Button(
                                "guanaco-7b",
                                elem_id="guanaco-7b",
                                elem_classes=["square"],
                            )
                            gr.Markdown("Guanaco", elem_classes=["center"])  
                            
                        with gr.Column(min_width=20):
                            falcon_7b = gr.Button(
                                "falcon-7b",
                                elem_id="falcon-7b",
                                elem_classes=["square"],
                            )
                            gr.Markdown("Falcon", elem_classes=["center"])    
                            
                        for _ in range(8):
                            with gr.Column(min_width=20, elem_classes=["placeholders"]):
                              _ = gr.Button("" ,elem_classes=["square"])
                              gr.Markdown("", elem_classes=["center"])
    
                        # with gr.Column(min_width=20):
                        #   stackllama7b = gr.Button("stackllama-7b", elem_id="stackllama-7b",elem_classes=["square"])
                        #   gr.Markdown("StackLLaMA", elem_classes=["center"])
                    #
                    gr.Markdown("## < 20B")
                    with gr.Row(elem_classes=["sub-container"]):
                        with gr.Column(min_width=20):
                            flan11b = gr.Button(
                                "flan-11b", elem_id="flan-11b", elem_classes=["square"]
                            )
                            gr.Markdown("Flan-XXL", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            koalpaca = gr.Button(
                                "koalpaca", elem_id="koalpaca", elem_classes=["square"]
                            )
                            gr.Markdown("koalpaca", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            alpaca_lora13b = alpaca_lora13b = gr.Button(
                                "alpaca-lora-13b",
                                elem_id="alpaca-lora-13b",
                                elem_classes=["square"],
                            )
                            gr.Markdown("Alpaca-LoRA", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            gpt4_alpaca_13b = gr.Button(
                                "gpt4-alpaca-13b",
                                elem_id="gpt4-alpaca-13b",
                                elem_classes=["square"],
                            )
                            gr.Markdown("GPT4-Alpaca-LoRA", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            stable_vicuna_13b = gr.Button(
                                "stable-vicuna-13b",
                                elem_id="stable-vicuna-13b",
                                elem_classes=["square"],
                            )
                            gr.Markdown("Stable-Vicuna", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            starchat_15b = gr.Button(
                                "starchat-15b",
                                elem_id="starchat-15b",
                                elem_classes=["square"],
                            )
                            gr.Markdown("StarChat", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            vicuna_13b = gr.Button(
                                "vicuna-13b", elem_id="vicuna-13b", elem_classes=["square"]
                            )
                            gr.Markdown("Vicuna", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            evolinstruct_vicuna_13b = gr.Button(
                                "evolinstruct-vicuna-13b",
                                elem_id="evolinstruct-vicuna-13b",
                                elem_classes=["square"],
                            )
                            gr.Markdown("EvolInstruct Vicuna", elem_classes=["center"])
    
                        with gr.Column(min_width=20):                        
                            baize_13b = gr.Button(
                                "baize-13b",
                                elem_id="baize-13b",
                                elem_classes=["square"],
                            )
                            gr.Markdown("Baize", elem_classes=["center"])                          
                            
                        with gr.Column(min_width=20):
                            guanaco_13b = gr.Button(
                                "guanaco-13b",
                                elem_id="guanaco-13b",
                                elem_classes=["square"],
                            )
                            gr.Markdown("Guanaco", elem_classes=["center"])                          
    
                        for _ in range(2):
                            with gr.Column(min_width=20, elem_classes=["placeholders"]):
                              _ = gr.Button("" ,elem_classes=["square"])
                              gr.Markdown("", elem_classes=["center"])               
    
                    gr.Markdown("## < 30B")
                    with gr.Row(elem_classes=["sub-container"]):
                        with gr.Column(min_width=20):
                            camel20b = gr.Button(
                                "camel-20b", elem_id="camel-20b", elem_classes=["square"]
                            )
                            gr.Markdown("Camel", elem_classes=["center"])
    
                        for _ in range(11):
                            with gr.Column(min_width=20, elem_classes=["placeholders"]):
                              _ = gr.Button("" ,elem_classes=["square"])
                              gr.Markdown("", elem_classes=["center"])

                    gr.Markdown("## < 40B")
                    with gr.Row(elem_classes=["sub-container"]):
                        with gr.Column(min_width=20):
                            guanaco_33b = gr.Button(
                                "guanaco-33b",
                                elem_id="guanaco-33b",
                                elem_classes=["square"],
                            )
                            gr.Markdown("Guanaco", elem_classes=["center"])                           
                        
                        with gr.Column(min_width=20):
                            falcon_40b = gr.Button(
                                "falcon-40b",
                                elem_id="falcon-40b",
                                elem_classes=["square"],
                            )
                            gr.Markdown("Falcon", elem_classes=["center"])
                            
                        for _ in range(10):
                            with gr.Column(min_width=20, elem_classes=["placeholders"]):
                              _ = gr.Button("" ,elem_classes=["square"])
                              gr.Markdown("", elem_classes=["center"])                          
                            
                    progress_view = gr.Textbox(label="Progress")
    
        with gr.Column(visible=False) as model_review_view:
            gr.Markdown("# Confirm the chosen model", elem_classes=["center"])
            with gr.Column(elem_id="container2"):
                with gr.Row():
                    model_image = gr.Image(None, interactive=False, show_label=False)
                    with gr.Column():
                        model_name = gr.Markdown("**Model name**")
                        model_params = gr.Markdown("Parameters\n: ...")
                        model_base = gr.Markdown("🤗 Hub(base)\n: ...")
                        model_ckpt = gr.Markdown("🤗 Hub(ckpt)\n: ...")
    
                with gr.Column():
                    gen_config_path = gr.Dropdown(
                        response_configs,
                        value=response_configs[0],
                        interactive=True,
                        label="Gen Config(response)",
                    )
                    gen_config_sum_path = gr.Dropdown(
                        summarization_configs,
                        value=summarization_configs[0],
                        interactive=True,
                        label="Gen Config(summarization)",
                        visible=False,
                    )
                    with gr.Row():
                        multi_gpu = gr.Checkbox(label="Multi GPU / (Non 8Bit mode)")
                        force_redownload = gr.Checkbox(label="Force Re-download")
    
                with gr.Row():
                    back_to_model_choose_btn = gr.Button("Back")
                    confirm_btn = gr.Button("Confirm")
    
                with gr.Column():
                    txt_view = gr.Textbox(label="Status")
                    progress_view2 = gr.Textbox(label="Progress")
    
        with gr.Column(visible=False) as chat_view:
            idx = gr.State(0)
            chat_state = gr.State()
            local_data = gr.JSON({}, visible=False)
    
            with gr.Row():
                with gr.Column(scale=1, min_width=180):
                    gr.Markdown("GradioChat", elem_id="left-top")
    
                    with gr.Column(elem_id="left-pane"):
                        with gr.Accordion("Histories", elem_id="chat-history-accordion", open=False):
                            channel_btns.append(gr.Button(channels[0], elem_classes=["custom-btn-highlight"]))
    
                            for channel in channels[1:]:
                                channel_btns.append(gr.Button(channel, elem_classes=["custom-btn"]))
    
                with gr.Column(scale=8, elem_id="right-pane"):
                    with gr.Column(
                        elem_id="initial-popup", visible=False
                    ) as example_block:
                        with gr.Row(scale=1):
                            with gr.Column(elem_id="initial-popup-left-pane"):
                                gr.Markdown("GradioChat", elem_id="initial-popup-title")
                                gr.Markdown(
                                    "Making the community's best AI chat models available to everyone."
                                )
                            with gr.Column(elem_id="initial-popup-right-pane"):
                                gr.Markdown(
                                    "Chat UI is now open sourced on Hugging Face Hub"
                                )
                                gr.Markdown(
                                    "check out the [↗ repository](https://huggingface.co/spaces/chansung/test-multi-conv)"
                                )
    
                        with gr.Column(scale=1):
                            gr.Markdown("Examples")
                            with gr.Row():
                                for example in examples:
                                    ex_btns.append(gr.Button(example, elem_classes=["example-btn"]))
    
                    with gr.Column(elem_id="aux-btns-popup", visible=True):
                        with gr.Row():
                            stop = gr.Button("Stop", elem_classes=["aux-btn"])
                            regenerate = gr.Button("Regenerate", interactive=False, elem_classes=["aux-btn"])
                            clean = gr.Button("Clean", elem_classes=["aux-btn"])
    
                    with gr.Accordion("Context Inspector", elem_id="aux-viewer", open=False):
                        context_inspector = gr.Textbox(
                            "",
                            elem_id="aux-viewer-inspector",
                            label="",
                            lines=30,
                            max_lines=50,
                        )                        
                            
                    chatbot = gr.Chatbot(elem_id='chatbot')
                    instruction_txtbox = gr.Textbox(
                        placeholder="Ask anything", label="",
                        elem_id="prompt-txt"
                    )
    
            with gr.Accordion("Constrol Panel", open=False) as control_panel:
                with gr.Column():
                    with gr.Column():
                        gr.Markdown("#### GenConfig for **response** text generation")
                        with gr.Row():
                            res_temp = gr.Slider(0.0, 2.0, 0, step=0.1, label="temp", interactive=True)
                            res_topp = gr.Slider(0.0, 2.0, 0, step=0.1, label="top_p", interactive=True)
                            res_topk = gr.Slider(20, 1000, 0, step=1, label="top_k", interactive=True)
                            res_rpen = gr.Slider(0.0, 2.0, 0, step=0.1, label="rep_penalty", interactive=True)
                            res_mnts = gr.Slider(64, 2048, 0, step=1, label="new_tokens", interactive=True)                            
                            res_beams = gr.Slider(1, 4, 0, step=1, label="beams")
                            res_cache = gr.Radio([True, False], value=0, label="cache", interactive=True)
                            res_sample = gr.Radio([True, False], value=0, label="sample", interactive=True)
                            res_eosid = gr.Number(value=0, visible=False, precision=0)
                            res_padid = gr.Number(value=0, visible=False, precision=0)
    
                    with gr.Column(visible=False):
                        gr.Markdown("#### GenConfig for **summary** text generation")
                        with gr.Row():
                            sum_temp = gr.Slider(0.0, 2.0, 0, step=0.1, label="temp", interactive=True)
                            sum_topp = gr.Slider(0.0, 2.0, 0, step=0.1, label="top_p", interactive=True)
                            sum_topk = gr.Slider(20, 1000, 0, step=1, label="top_k", interactive=True)
                            sum_rpen = gr.Slider(0.0, 2.0, 0, step=0.1, label="rep_penalty", interactive=True)
                            sum_mnts = gr.Slider(64, 2048, 0, step=1, label="new_tokens", interactive=True)
                            sum_beams = gr.Slider(1, 8, 0, step=1, label="beams", interactive=True)
                            sum_cache = gr.Radio([True, False], value=0, label="cache", interactive=True)
                            sum_sample = gr.Radio([True, False], value=0, label="sample", interactive=True)
                            sum_eosid = gr.Number(value=0, visible=False, precision=0)
                            sum_padid = gr.Number(value=0, visible=False, precision=0)
    
                    with gr.Column(visible=False):
                        gr.Markdown("#### Context managements")
                        with gr.Row():
                            ctx_num_lconv = gr.Slider(2, 6, 3, step=1, label="num of last talks to keep", interactive=True)
                            ctx_sum_prompt = gr.Textbox(
                                "summarize our conversations. what have we discussed about so far?",
                                label="design a prompt to summarize the conversations"
                            )
    
            btns = [
                t5_vicuna_3b, flan3b, camel5b, alpaca_lora7b, stablelm7b,
                gpt4_alpaca_7b, os_stablelm7b, mpt_7b, redpajama_7b, llama_deus_7b, 
                evolinstruct_vicuna_7b, alpacoom_7b, baize_7b, guanaco_7b,
                falcon_7b,
                flan11b, koalpaca, alpaca_lora13b, gpt4_alpaca_13b, stable_vicuna_13b,
                starchat_15b, vicuna_7b, vicuna_13b, evolinstruct_vicuna_13b, baize_13b, guanaco_13b,
                camel20b,
                guanaco_33b, falcon_40b,
            ]
            for btn in btns:
                btn.click(
                    move_to_second_view,
                    btn,
                    [
                        model_choice_view, model_review_view,
                        model_image, model_name, model_params, model_base, model_ckpt,
                        progress_view
                    ]
                )
    
            back_to_model_choose_btn.click(
                move_to_first_view,
                None,
                [model_choice_view, model_review_view]
            )
    
            confirm_btn.click(
                lambda: "Start downloading/loading the model...", None, txt_view
            ).then(
                download_completed,
                [model_name, model_base, model_ckpt, gen_config_path, gen_config_sum_path, multi_gpu, force_redownload],
                [progress_view2]
            ).then(
                lambda: "Model is fully loaded...", None, txt_view
            ).then(
                lambda: time.sleep(2), None, None
            ).then(
                move_to_third_view,
                None,
                [progress_view2, model_review_view, chat_view, chatbot, chat_state,
                res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
                sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid]
            )                        
             
            for btn in channel_btns:
                btn.click(
                    set_chatbot,
                    [btn, local_data, chat_state],
                    [chatbot, idx, example_block, regenerate]
                ).then(
                    None, btn, None, 
                    _js=UPDATE_LEFT_BTNS_STATE        
                )
            
            for btn in ex_btns:
                btn.click(
                    set_example,
                    [btn],
                    [instruction_txtbox, example_block]  
                )
    
            instruction_txtbox.submit(
                lambda: [
                    gr.update(visible=False),
                    gr.update(interactive=True)
                ],
                None,
                [example_block, regenerate]
            )
            
            send_event = instruction_txtbox.submit(
                central.chat_stream,
                [idx, local_data, instruction_txtbox, chat_state,
                ctx_num_lconv, ctx_sum_prompt,
                res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
                sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid],
                [instruction_txtbox, chatbot, context_inspector, local_data],
            )
            
            instruction_txtbox.submit(
                None, local_data, None, 
                _js="(v)=>{ setStorage('local_data',v) }"
            )
    
            regenerate.click(
                rollback_last,
                [idx, local_data, chat_state],
                [instruction_txtbox, chatbot, local_data, regenerate]
            ).then(
                central.chat_stream,
                [idx, local_data, instruction_txtbox, chat_state,
                ctx_num_lconv, ctx_sum_prompt,
                res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
                sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid],
                [instruction_txtbox, chatbot, context_inspector, local_data],            
            ).then(
                lambda: gr.update(interactive=True),
                None,
                regenerate
            ).then(
                None, local_data, None, 
                _js="(v)=>{ setStorage('local_data',v) }"  
            )
            
            stop.click(
                None, None, None,
                cancels=[send_event]
            )
    
            clean.click(
                reset_chat,
                [idx, local_data, chat_state],
                [instruction_txtbox, chatbot, local_data, example_block, regenerate]
            ).then(
                None, local_data, None, 
                _js="(v)=>{ setStorage('local_data',v) }"
            )
            
            demo.load(
              None,
              inputs=None,
              outputs=[chatbot, local_data],
              _js=GET_LOCAL_STORAGE,
            )          
            
    demo.queue().launch(
        server_port=6006, server_name="0.0.0.0", debug=True,
        root_path=f"{root_path}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-path', default="")
    args = parser.parse_args()
    
    main(args.root_path)
