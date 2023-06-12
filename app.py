import time
import json
import copy
import types
from os import listdir
from os.path import isfile, join
import argparse
import gradio as gr
import global_vars
from chats import central
from transformers import AutoModelForCausalLM
from miscs.styles import MODEL_SELECTION_CSS
from miscs.js import GET_LOCAL_STORAGE, UPDATE_LEFT_BTNS_STATE
from utils import get_chat_manager, get_global_context

from pingpong.pingpong import PingPong
from pingpong.gradio import GradioAlpacaChatPPManager
from pingpong.gradio import GradioKoAlpacaChatPPManager
from pingpong.gradio import GradioStableLMChatPPManager
from pingpong.gradio import GradioFlanAlpacaChatPPManager
from pingpong.gradio import GradioOSStableLMChatPPManager
from pingpong.gradio import GradioVicunaChatPPManager
from pingpong.gradio import GradioStableVicunaChatPPManager
from pingpong.gradio import GradioStarChatPPManager
from pingpong.gradio import GradioMPTChatPPManager
from pingpong.gradio import GradioRedPajamaChatPPManager
from pingpong.gradio import GradioBaizeChatPPManager

# no cpu for 
# - falcon families (too slow)

load_mode_list = ["cpu"]

ex_file = open("examples.txt", "r")
examples = ex_file.read().split("\n")
ex_btns = []

chl_file = open("channels.txt", "r")
channels = chl_file.read().split("\n")
channel_btns = []

default_ppm = GradioAlpacaChatPPManager()
default_ppm.ctx = "Context at top"
default_ppm.pingpongs = [
    PingPong("user input #1...", "bot response #1..."),
    PingPong("user input #2...", "bot response #2..."),
]
chosen_ppm = copy.deepcopy(default_ppm)

prompt_styles = {
    "Alpaca": default_ppm,
    "Baize": GradioBaizeChatPPManager(),
    "Koalpaca": GradioKoAlpacaChatPPManager(),
    "MPT": GradioMPTChatPPManager(),
    "OpenAssistant StableLM": GradioOSStableLMChatPPManager(),
    "RedPajama": GradioRedPajamaChatPPManager(),
    "StableVicuna": GradioVicunaChatPPManager(),
    "StableLM": GradioStableLMChatPPManager(),
    "StarChat": GradioStarChatPPManager(),
    "Vicuna": GradioVicunaChatPPManager(),
}

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

###

def move_to_model_select_view():
    return (
        "move to model select view",
        gr.update(visible=False),
        gr.update(visible=True),
    )
    
def use_chosen_model():
    try:
        test = global_vars.model
    except AttributeError:
        raise gr.Error("There is no previously chosen model")

    gen_config = global_vars.gen_config
    gen_sum_config = global_vars.gen_config_summarization

    if global_vars.model_type == "custom":
        ppmanager_type = chosen_ppm
    else:
        ppmanager_type = get_chat_manager(global_vars.model_type)

    return (
        "Preparation done!",
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(label=global_vars.model_type),
        {
            "ppmanager_type": ppmanager_type,
            "model_type": global_vars.model_type,
        },
        get_global_context(global_vars.model_type),
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
    
def move_to_byom_view():
    load_mode_list = []
    if global_vars.cuda_availability:
        load_mode_list.extend(["gpu(half)", "gpu(load_in_8bit)", "gpu(load_in_4bit)"])

    if global_vars.mps_availability:
        load_mode_list.append("apple silicon")
        
    load_mode_list.append("cpu")
    
    return (
        "move to the byom view",
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(choices=load_mode_list, value=load_mode_list[0])
    )

def prompt_style_change(key):
    ppm = prompt_styles[key]
    ppm.ctx = "Context at top"
    ppm.pingpongs = [
        PingPong("user input #1...", "bot response #1..."),
        PingPong("user input #2...", "bot response #2..."),
    ]
    chosen_ppm = copy.deepcopy(ppm)
    chosen_ppm.ctx = ""
    chosen_ppm.pingpongs = []
    
    return ppm.build_prompts()

def byom_load(
    base, ckpt, model_cls, tokenizer_cls,
    bos_token_id, eos_token_id, pad_token_id, 
    load_mode,
):
    
    # mode_cpu, model_mps, mode_8bit, mode_4bit, mode_full_gpu
    global_vars.initialize_globals_byom(
        base, ckpt, model_cls, tokenizer_cls,
        bos_token_id, eos_token_id, pad_token_id, 
        True if load_mode == "cpu" else False,
        True if load_mode == "apple silicon" else False,
        True if load_mode == "8bit" else False,
        True if load_mode == "4bit" else False,
        True if load_mode == "gpu(half)" else False
    )
    
    return (
        ""
    )
    
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

    guard_vram = 5 * 1024.
    vram_req_full = int(info["vram(full)"]) + guard_vram
    vram_req_8bit = int(info["vram(8bit)"]) + guard_vram
    vram_req_4bit = int(info["vram(4bit)"]) + guard_vram
    
    load_mode_list = []
    
    if global_vars.cuda_availability:
        print(f"total vram = {global_vars.available_vrams_mb}")
        print(f"required vram(full={info['vram(full)']}, 8bit={info['vram(8bit)']}, 4bit={info['vram(4bit)']})")
        
        if global_vars.available_vrams_mb >= vram_req_full:
            load_mode_list.append("gpu(half)")
            
        if global_vars.available_vrams_mb >= vram_req_8bit:
            load_mode_list.append("gpu(load_in_8bit)")
            
        if global_vars.available_vrams_mb >= vram_req_4bit:
            load_mode_list.append("gpu(load_in_4bit)")

    if global_vars.mps_availability:
        load_mode_list.append("apple silicon")

    load_mode_list.extend(["cpu"])
    
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        info["thumb"],
        f"## {btn}",
        f"**Parameters**\n: Approx. {info['parameters']}",
        f"**🤗 Hub(base)**\n: {info['hub(base)']}",
        f"**🤗 Hub(LoRA)**\n: {info['hub(ckpt)']}",
        info['desc'],
        f"""**Min VRAM requirements** :
|             half precision            |             load_in_8bit           |              load_in_4bit          | 
| ------------------------------------- | ---------------------------------- | ---------------------------------- | 
|   {round(vram_req_full/1024., 1)}GiB  | {round(vram_req_8bit/1024., 1)}GiB | {round(vram_req_4bit/1024., 1)}GiB |
""",
        info['default_gen_config'],
        info['example1'],
        info['example2'],
        info['example3'],
        info['example4'],
        info['thumb-tiny'],        
        gr.update(choices=load_mode_list, value=load_mode_list[0]),
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
    load_mode,
    thumbnail_tiny,
    force_download,
):
    tmp_args = types.SimpleNamespace()
    tmp_args.base_url = model_base.split(":")[-1].split("</p")[0].strip()
    tmp_args.ft_ckpt_url = model_ckpt.split(":")[-1].split("</p")[0].strip()
    tmp_args.gen_config_path = gen_config_path
    tmp_args.gen_config_summarization_path = gen_config_sum_path
    tmp_args.force_download_ckpt = force_download
    tmp_args.thumbnail_tiny = thumbnail_tiny
    
    tmp_args.mode_cpu = True if load_mode == "cpu" else False
    tmp_args.mode_mps = True if load_mode == "apple silicon" else False
    tmp_args.mode_8bit = True if load_mode == "gpu(load_in_8bit)" else False
    tmp_args.mode_4bit = True if load_mode == "gpu(load_in_4bit)" else False
    tmp_args.mode_full_gpu = True if load_mode == "gpu(half)" else False
    
    try:
        global_vars.initialize_globals(tmp_args)
    except RuntimeError as e:
        raise gr.Error("GPU memory is not enough to load this model.")
        
    return "Download completed!"

def move_to_third_view():  
    gen_config = global_vars.gen_config
    gen_sum_config = global_vars.gen_config_summarization

    if global_vars.model_type == "custom":
        ppmanager_type = chosen_ppm
    else:
        ppmanager_type = get_chat_manager(global_vars.model_type)

    return (
        "Preparation done!",
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(label=global_vars.model_type),
        {
            "ppmanager_type": ppmanager_type,
            "model_type": global_vars.model_type,
        },
        get_global_context(global_vars.model_type),
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

def main(args):
    with gr.Blocks(css=MODEL_SELECTION_CSS, theme='gradio/soft') as demo:
        with gr.Column(visible=True, elem_id="landing-container") as landing_view:
            gr.Markdown("# Chat with LLM", elem_classes=["center"])
            with gr.Row(elem_id="landing-container-selection"):
                with gr.Column():
                    gr.Markdown("""This is the landing page of the project, [LLM As Chatbot](https://github.com/deep-diver/LLM-As-Chatbot). This appliction is designed for personal use only. A single model will be selected at a time even if you open up a new browser or a tab. As an initial choice, please select one of the following menu""")

                    gr.Markdown("""      
**Bring your own model**: You can chat with arbitrary models. If your own custom model is based on 🤗 Hugging Face's [transformers](https://huggingface.co/docs/transformers/index) library, you will propbably be able to bring it into this application with this menu

**Select a model from model pool**: You can chat with one of the popular open source Large Language Model

**Use currently selected model**: If you have already selected, but if you came back to this landing page accidently, you can directly go back to the chatting mode with this menu                    
""")                    
                    
                    byom = gr.Button("🫵🏼 Bring your own model", elem_id="go-byom-select", elem_classes=["square", "landing-btn"])
                    select_model = gr.Button("🦙 Select a model from model pool", elem_id="go-model-select", elem_classes=["square", "landing-btn"])
                    chosen_model = gr.Button("↪️ Use currently selected model", elem_id="go-use-selected-model", elem_classes=["square", "landing-btn"])

                    with gr.Column(elem_id="landing-bottom"):
                        progress_view0 = gr.Textbox(label="Progress", elem_classes=["progress-view"])
                        gr.Markdown("""[project](https://github.com/deep-diver/LLM-As-Chatbot)
[developer](https://github.com/deep-diver)
""", elem_classes=["center"])
    
        with gr.Column(visible=False) as model_choice_view:
            gr.Markdown("# Choose a Model", elem_classes=["center"])
            with gr.Row(elem_id="container"):
                with gr.Column():
                    gr.Markdown("## ~ 10B Parameters")
                    with gr.Row(elem_classes=["sub-container"]):
                        with gr.Column(min_width=20):
                            t5_vicuna_3b = gr.Button("t5-vicuna-3b", elem_id="t5-vicuna-3b", elem_classes=["square"])
                            gr.Markdown("T5 Vicuna", elem_classes=["center"])
    
                        with gr.Column(min_width=20, visible=False):
                            flan3b = gr.Button("flan-3b", elem_id="flan-3b", elem_classes=["square"])
                            gr.Markdown("Flan-XL", elem_classes=["center"])

                        # with gr.Column(min_width=20):
                        #     replit_3b = gr.Button("replit-3b", elem_id="replit-3b", elem_classes=["square"])
                        #     gr.Markdown("Replit Instruct", elem_classes=["center"])                        
                        
                        with gr.Column(min_width=20):
                            camel5b = gr.Button("camel-5b", elem_id="camel-5b", elem_classes=["square"])
                            gr.Markdown("Camel", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            alpaca_lora7b = gr.Button("alpaca-lora-7b", elem_id="alpaca-lora-7b", elem_classes=["square"])
                            gr.Markdown("Alpaca-LoRA", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            stablelm7b = gr.Button("stablelm-7b", elem_id="stablelm-7b", elem_classes=["square"])
                            gr.Markdown("StableLM", elem_classes=["center"])
    
                        with gr.Column(min_width=20, visible=False):
                            os_stablelm7b = gr.Button("os-stablelm-7b", elem_id="os-stablelm-7b", elem_classes=["square"])
                            gr.Markdown("OA+StableLM", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            gpt4_alpaca_7b = gr.Button("gpt4-alpaca-7b", elem_id="gpt4-alpaca-7b", elem_classes=["square"])
                            gr.Markdown("GPT4-Alpaca-LoRA", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            mpt_7b = gr.Button("mpt-7b", elem_id="mpt-7b", elem_classes=["square"])
                            gr.Markdown("MPT", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            redpajama_7b = gr.Button("redpajama-7b", elem_id="redpajama-7b", elem_classes=["square"])
                            gr.Markdown("RedPajama", elem_classes=["center"])
                    
                        with gr.Column(min_width=20, visible=False):
                            redpajama_instruct_7b = gr.Button("redpajama-instruct-7b", elem_id="redpajama-instruct-7b", elem_classes=["square"])
                            gr.Markdown("RedPajama Instruct", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            vicuna_7b = gr.Button("vicuna-7b", elem_id="vicuna-7b", elem_classes=["square"])
                            gr.Markdown("Vicuna", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            llama_deus_7b = gr.Button("llama-deus-7b", elem_id="llama-deus-7b",elem_classes=["square"])
                            gr.Markdown("LLaMA Deus", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            evolinstruct_vicuna_7b = gr.Button("evolinstruct-vicuna-7b", elem_id="evolinstruct-vicuna-7b", elem_classes=["square"])
                            gr.Markdown("EvolInstruct Vicuna", elem_classes=["center"])
    
                        with gr.Column(min_width=20, visible=False):
                            alpacoom_7b = gr.Button("alpacoom-7b", elem_id="alpacoom-7b", elem_classes=["square"])
                            gr.Markdown("Alpacoom", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            baize_7b = gr.Button("baize-7b", elem_id="baize-7b", elem_classes=["square"])
                            gr.Markdown("Baize", elem_classes=["center"])                        
                            
                        with gr.Column(min_width=20):
                            guanaco_7b = gr.Button("guanaco-7b", elem_id="guanaco-7b", elem_classes=["square"])
                            gr.Markdown("Guanaco", elem_classes=["center"])  
                            
                        with gr.Column(min_width=20):
                            falcon_7b = gr.Button("falcon-7b", elem_id="falcon-7b", elem_classes=["square"])
                            gr.Markdown("Falcon", elem_classes=["center"])

                        with gr.Column(min_width=20):
                            wizard_falcon_7b = gr.Button("wizard-falcon-7b", elem_id="wizard-falcon-7b", elem_classes=["square"])
                            gr.Markdown("Wizard Falcon", elem_classes=["center"])
                            
                        with gr.Column(min_width=20):
                            airoboros_7b = gr.Button("airoboros-7b", elem_id="airoboros-7b", elem_classes=["square"])
                            gr.Markdown("Airoboros", elem_classes=["center"])
                            
                        with gr.Column(min_width=20):
                            samantha_7b = gr.Button("samantha-7b", elem_id="samantha-7b", elem_classes=["square"])
                            gr.Markdown("Samantha", elem_classes=["center"])

                    gr.Markdown("## ~ 20B Parameters")
                    with gr.Row(elem_classes=["sub-container"]):
                        with gr.Column(min_width=20, visible=False):
                            flan11b = gr.Button("flan-11b", elem_id="flan-11b", elem_classes=["square"])
                            gr.Markdown("Flan-XXL", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            koalpaca = gr.Button("koalpaca", elem_id="koalpaca", elem_classes=["square"])
                            gr.Markdown("koalpaca", elem_classes=["center"])

                        with gr.Column(min_width=20):
                            kullm = gr.Button("kullm", elem_id="kullm", elem_classes=["square"])
                            gr.Markdown("KULLM", elem_classes=["center"])
                        
                        with gr.Column(min_width=20):
                            alpaca_lora13b = gr.Button("alpaca-lora-13b", elem_id="alpaca-lora-13b", elem_classes=["square"])
                            gr.Markdown("Alpaca-LoRA", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            gpt4_alpaca_13b = gr.Button("gpt4-alpaca-13b", elem_id="gpt4-alpaca-13b", elem_classes=["square"])
                            gr.Markdown("GPT4-Alpaca-LoRA", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            stable_vicuna_13b = gr.Button("stable-vicuna-13b", elem_id="stable-vicuna-13b", elem_classes=["square"])
                            gr.Markdown("Stable-Vicuna", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            starchat_15b = gr.Button("starchat-15b", elem_id="starchat-15b", elem_classes=["square"])
                            gr.Markdown("StarChat", elem_classes=["center"])

                        with gr.Column(min_width=20):
                            starchat_beta_15b = gr.Button("starchat-beta-15b", elem_id="starchat-beta-15b", elem_classes=["square"])
                            gr.Markdown("StarChat β", elem_classes=["center"])

                        with gr.Column(min_width=20):
                            vicuna_13b = gr.Button("vicuna-13b", elem_id="vicuna-13b", elem_classes=["square"])
                            gr.Markdown("Vicuna", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            evolinstruct_vicuna_13b = gr.Button("evolinstruct-vicuna-13b", elem_id="evolinstruct-vicuna-13b", elem_classes=["square"])
                            gr.Markdown("EvolInstruct Vicuna", elem_classes=["center"])
    
                        with gr.Column(min_width=20):
                            baize_13b = gr.Button("baize-13b", elem_id="baize-13b", elem_classes=["square"])
                            gr.Markdown("Baize", elem_classes=["center"])                          
                            
                        with gr.Column(min_width=20):
                            guanaco_13b = gr.Button("guanaco-13b", elem_id="guanaco-13b", elem_classes=["square"])
                            gr.Markdown("Guanaco", elem_classes=["center"])

                        with gr.Column(min_width=20):
                            nous_hermes_13b = gr.Button("nous-hermes-13b", elem_id="nous-hermes-13b", elem_classes=["square"])
                            gr.Markdown("Nous Hermes", elem_classes=["center"])
                            
                        with gr.Column(min_width=20):
                            airoboros_13b = gr.Button("airoboros-13b", elem_id="airoboros-13b", elem_classes=["square"])
                            gr.Markdown("Airoboros", elem_classes=["center"])
                            
                        with gr.Column(min_width=20):
                            samantha_13b = gr.Button("samantha-13b", elem_id="samantha-13b", elem_classes=["square"])
                            gr.Markdown("Samantha", elem_classes=["center"])
                            
                        with gr.Column(min_width=20):
                            chronos_13b = gr.Button("chronos-13b", elem_id="chronos-13b", elem_classes=["square"])
                            gr.Markdown("Chronos", elem_classes=["center"])
                            
                        with gr.Column(min_width=20):
                            wizardlm_13b = gr.Button("wizardlm-13b", elem_id="wizardlm-13b", elem_classes=["square"])
                            gr.Markdown("WizardLM", elem_classes=["center"])
                            
                    gr.Markdown("## ~ 30B Parameters", visible=False)
                    with gr.Row(elem_classes=["sub-container"], visible=False):
                        with gr.Column(min_width=20):
                            camel20b = gr.Button("camel-20b", elem_id="camel-20b", elem_classes=["square"])
                            gr.Markdown("Camel", elem_classes=["center"])

                    gr.Markdown("## ~ 40B Parameters")
                    with gr.Row(elem_classes=["sub-container"]):
                        with gr.Column(min_width=20):
                            guanaco_33b = gr.Button("guanaco-33b", elem_id="guanaco-33b", elem_classes=["square"])
                            gr.Markdown("Guanaco", elem_classes=["center"])                           
                        
                        with gr.Column(min_width=20):
                            falcon_40b = gr.Button("falcon-40b", elem_id="falcon-40b", elem_classes=["square"])
                            gr.Markdown("Falcon", elem_classes=["center"])

                        with gr.Column(min_width=20):
                            wizard_falcon_40b = gr.Button("wizard-falcon-40b", elem_id="wizard-falcon-40b", elem_classes=["square"])
                            gr.Markdown("Wizard Falcon", elem_classes=["center"])
                            
                        with gr.Column(min_width=20):
                            samantha_33b = gr.Button("samantha-33b", elem_id="samantha-33b", elem_classes=["square"])
                            gr.Markdown("Samantha", elem_classes=["center"])
                            
                        with gr.Column(min_width=20):
                            lazarus_30b = gr.Button("lazarus-30b", elem_id="lazarus-30b", elem_classes=["square"])
                            gr.Markdown("Lazarus", elem_classes=["center"])
                            
                        with gr.Column(min_width=20):
                            chronos_33b = gr.Button("chronos-33b", elem_id="chronos-33b", elem_classes=["square"])
                            gr.Markdown("Chronos", elem_classes=["center"])
                            
                        with gr.Column(min_width=20):
                            wizardlm_30b = gr.Button("wizardlm-30b", elem_id="wizardlm-30b", elem_classes=["square"])
                            gr.Markdown("WizardLM", elem_classes=["center"])                            

                    progress_view = gr.Textbox(label="Progress", elem_classes=["progress-view"])

        with gr.Column(visible=False) as byom_input_view:
            with gr.Column(elem_id="container3"):
                gr.Markdown("# Bring Your Own Model", elem_classes=["center"])
                
                gr.Markdown("### Model configuration")
                byom_base = gr.Textbox(label="Base", placeholder="Enter path or 🤗 hub ID of the base model", interactive=True)
                byom_ckpt = gr.Textbox(label="LoRA ckpt", placeholder="Enter path or 🤗 hub ID of the LoRA checkpoint", interactive=True)
                
                with gr.Accordion("Advanced options", open=False):
                    gr.Markdown("If you leave the below textboxes empty, `transformers.AutoModelForCausalLM` and `transformers.AutoTokenizer` classes will be used by default. If you need any specific class, please type them below.")
                    byom_model_cls = gr.Textbox(label="Base model class", placeholder="Enter base model class", interactive=True)
                    byom_tokenizer_cls = gr.Textbox(label="Base tokenizer class", placeholder="Enter base tokenizer class", interactive=True)

                    with gr.Column():
                        gr.Markdown("If you leave the below textboxes empty, any token ids for bos, eos, and pad will not be specified in `GenerationConfig`. If you think that you need to specify them. please type them below in decimal format.")                        
                        with gr.Row():
                            byom_bos_token_id = gr.Textbox(label="bos_token_id", placeholder="for GenConfig")
                            byom_eos_token_id = gr.Textbox(label="eos_token_id", placeholder="for GenConfig")
                            byom_pad_token_id = gr.Textbox(label="pad_token_id", placeholder="for GenConfig")
                    
                    with gr.Row():
                        byom_load_mode = gr.Radio(
                            load_mode_list,
                            value=load_mode_list[0],
                            label="load mode",
                            elem_classes=["load-mode-selector"]
                        )                        
                
                gr.Markdown("### Prompt configuration")
                prompt_style_selector = gr.Dropdown(
                    label="Prompt style", 
                    interactive=True,
                    choices=list(prompt_styles.keys()), 
                    value="Alpaca"
                )
                with gr.Accordion("Prompt style preview", open=False):
                    prompt_style_previewer = gr.Textbox(
                        label="How prompt is actually structured",
                        lines=16,
                        value=default_ppm.build_prompts())
                    
                with gr.Row():
                    byom_back_btn = gr.Button("Back")
                    byom_confirm_btn = gr.Button("Confirm")

                with gr.Column(elem_classes=["progress-view"]):
                    txt_view3 = gr.Textbox(label="Status")
                    progress_view3 = gr.Textbox(label="Progress")
        
        with gr.Column(visible=False) as model_review_view:
            gr.Markdown("# Confirm the chosen model", elem_classes=["center"])

            with gr.Column(elem_id="container2"):
                gr.Markdown("Please expect loading time to be longer than expected. Depending on the size of models, it will probably take from 100 to 300 seconds or so. Especially, expect the longest loading time with MPT model.")

                with gr.Row():
                    model_image = gr.Image(None, interactive=False, show_label=False)
                    with gr.Column():
                        model_name = gr.Markdown("**Model name**")
                        model_desc = gr.Markdown("...")                        
                        model_params = gr.Markdown("Parameters\n: ...")             
                        model_base = gr.Markdown("🤗 Hub(base)\n: ...")
                        model_ckpt = gr.Markdown("🤗 Hub(LoRA)\n: ...")
                        model_vram = gr.Markdown(f"""**Minimal VRAM requirement** :
|          half precision        |        load_in_8bit       |         load_in_4bit      | 
| ------------------------------ | ------------------------- | ------------------------- | 
|   {round(7830/1024., 1)}GiB    | {round(5224/1024., 1)}GiB | {round(4324/1024., 1)}GiB |
""")
                        model_thumbnail_tiny = gr.Textbox("", visible=False)
    
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
                        load_mode = gr.Radio(
                            load_mode_list,
                            value=load_mode_list[0],
                            label="load mode",
                            elem_classes=["load-mode-selector"]
                        )
                        force_redownload = gr.Checkbox(label="Force Re-download", interactive=False, visible=False)

                    with gr.Accordion("Example showcases", open=False):
                        with gr.Tab("Ex1"):
                            example_showcase1 = gr.Chatbot(
                                [("hello", "world"), ("damn", "good")]
                            )
                        with gr.Tab("Ex2"):
                            example_showcase2 = gr.Chatbot(
                                [("hello", "world"), ("damn", "good")]
                            )
                        with gr.Tab("Ex3"):
                            example_showcase3 = gr.Chatbot(
                                [("hello", "world"), ("damn", "good")]
                            )
                        with gr.Tab("Ex4"):
                            example_showcase4 = gr.Chatbot(
                                [("hello", "world"), ("damn", "good")]
                            )
                
                with gr.Row():
                    back_to_model_choose_btn = gr.Button("Back")
                    confirm_btn = gr.Button("Confirm")
    
                with gr.Column(elem_classes=["progress-view"]):
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
                        chat_back_btn = gr.Button("Back", elem_id="chat-back-btn")
                        
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
                                gr.Markdown("Making the community's best AI chat models available to everyone.")
                            with gr.Column(elem_id="initial-popup-right-pane"):
                                gr.Markdown("Chat UI is now open sourced on Hugging Face Hub")
                                gr.Markdown("check out the [↗ repository](https://huggingface.co/spaces/chansung/test-multi-conv)")
    
                        with gr.Column(scale=1):
                            gr.Markdown("Examples")
                            with gr.Row():
                                for example in examples:
                                    ex_btns.append(gr.Button(example, elem_classes=["example-btn"]))
    
                    with gr.Column(elem_id="aux-btns-popup", visible=True):
                        with gr.Row():
                            stop = gr.Button("Stop", elem_classes=["aux-btn"])
                            regenerate = gr.Button("Regen", interactive=False, elem_classes=["aux-btn"])
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
                    instruction_txtbox = gr.Textbox(placeholder="Ask anything", label="", elem_id="prompt-txt")
    
            with gr.Accordion("Control Panel", open=False) as control_panel:
                with gr.Column():
                    with gr.Column():
                        gr.Markdown("#### Global context")
                        with gr.Accordion("global context will persist during conversation, and it is placed at the top of the prompt", open=False):
                            global_context = gr.Textbox(
                                "global context",
                                lines=5,
                                max_lines=10,
                                interactive=True,
                                elem_id="global-context"
                            )
                        
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
    
                    with gr.Column():
                        gr.Markdown("#### Context managements")
                        with gr.Row():
                            ctx_num_lconv = gr.Slider(2, 10, 3, step=1, label="number of recent talks to keep", interactive=True)
                            ctx_sum_prompt = gr.Textbox(
                                "summarize our conversations. what have we discussed about so far?",
                                label="design a prompt to summarize the conversations",
                                visible=False
                            )
    
            btns = [
                t5_vicuna_3b, flan3b, camel5b, alpaca_lora7b, stablelm7b,
                gpt4_alpaca_7b, os_stablelm7b, mpt_7b, redpajama_7b, redpajama_instruct_7b, llama_deus_7b, 
                evolinstruct_vicuna_7b, alpacoom_7b, baize_7b, guanaco_7b,
                falcon_7b, wizard_falcon_7b, airoboros_7b, samantha_7b,
                flan11b, koalpaca, kullm, alpaca_lora13b, gpt4_alpaca_13b, stable_vicuna_13b,
                starchat_15b, starchat_beta_15b, vicuna_7b, vicuna_13b, evolinstruct_vicuna_13b, 
                baize_13b, guanaco_13b, nous_hermes_13b, airoboros_13b, samantha_13b, chronos_13b,
                wizardlm_13b,
                camel20b,
                guanaco_33b, falcon_40b, wizard_falcon_40b, samantha_33b, lazarus_30b, chronos_33b,
                wizardlm_30b,
            ]
            for btn in btns:
                btn.click(
                    move_to_second_view,
                    btn,
                    [
                        model_choice_view, model_review_view,
                        model_image, model_name, model_params, model_base, model_ckpt,
                        model_desc, model_vram, gen_config_path, 
                        example_showcase1, example_showcase2, example_showcase3, example_showcase4,
                        model_thumbnail_tiny, load_mode, 
                        progress_view
                    ]
                )

            select_model.click(
                move_to_model_select_view,
                None,
                [progress_view0, landing_view, model_choice_view]
            )
            
            chosen_model.click(
                use_chosen_model,
                None,
                [progress_view0, landing_view, chat_view, chatbot, chat_state, global_context,
                res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
                sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid]
            )
          
            byom.click(
                move_to_byom_view,
                None,
                [progress_view0, landing_view, byom_input_view, byom_load_mode]
            )

            byom_back_btn.click(
                move_to_first_view,
                None,
                [landing_view, byom_input_view]
            )

            byom_confirm_btn.click(
                lambda: "Start downloading/loading the model...", None, txt_view3
            ).then(
                byom_load,
                [byom_base, byom_ckpt, byom_model_cls, byom_tokenizer_cls,
                byom_bos_token_id, byom_eos_token_id, byom_pad_token_id, 
                byom_load_mode],
                [progress_view3]
            ).then(
                lambda: "Model is fully loaded...", None, txt_view3
            ).then(
                move_to_third_view,
                None,
                [progress_view3, byom_input_view, chat_view, chatbot, chat_state, global_context,
                res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
                sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid]
            )

            prompt_style_selector.change(
                prompt_style_change,
                prompt_style_selector,
                prompt_style_previewer
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
                [model_name, model_base, model_ckpt, gen_config_path, gen_config_sum_path, load_mode, model_thumbnail_tiny, force_redownload],
                [progress_view2]
            ).then(
                lambda: "Model is fully loaded...", None, txt_view
            ).then(
                lambda: time.sleep(2), None, None
            ).then(
                move_to_third_view,
                None,
                [progress_view2, model_review_view, chat_view, chatbot, chat_state, global_context,
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
                global_context, ctx_num_lconv, ctx_sum_prompt,
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
                global_context, ctx_num_lconv, ctx_sum_prompt,
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

            chat_back_btn.click(
                lambda: [gr.update(visible=False), gr.update(visible=True)],
                None,
                [chat_view, landing_view]
            )
          
            demo.load(
              None,
              inputs=None,
              outputs=[chatbot, local_data],
              _js=GET_LOCAL_STORAGE,
            )          
            
    demo.queue().launch(
        server_port=6006, 
        server_name="0.0.0.0", 
        debug=args.debug,
        share=args.share,
        root_path=f"{args.root_path}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-path', default="")
    parser.add_argument('--share', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--debug', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    main(args)
