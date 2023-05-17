import time
import json
from os import listdir
from os.path import isfile, join
import gradio as gr
import args
import global_vars
from chats import central
from transformers import AutoModelForCausalLM
from miscs.styles import MODEL_SELECTION_CSS
from utils import get_chat_interface, get_chat_manager

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
    ""
  )

def move_to_first_view():
  return (
    gr.update(visible=True),
    gr.update(visible=False)
  )

def download_completed(
  model_name, model_base, model_ckpt,
  gen_config_path, gen_config_sum_path,
  multi_gpu, chat_only_mode):

  tmp_args = args.parse_args()
  tmp_args.base_url = model_base.split(":")[-1].split("</p")[0].strip()
  tmp_args.ft_ckpt_url = model_ckpt.split(":")[-1].split("</p")[0].strip()
  tmp_args.gen_config_path = gen_config_path
  tmp_args.gen_config_summarization_path = gen_config_sum_path
  tmp_args.multi_gpu = multi_gpu
  tmp_args.chat_only_mode = chat_only_mode

  global_vars.initialize_globals(tmp_args)
  return (
    "Download completed!", 
    gr.update(visible=not tmp_args.chat_only_mode),
    gr.update(visible=not tmp_args.chat_only_mode)
  )

def move_to_third_view():
  gen_config = global_vars.gen_config
  gen_sum_config = global_vars.gen_config_summarization

  return (
    "Preparation done!", 
    gr.update(visible=False),
    gr.update(visible=True),
    gr.update(label=global_vars.model_type),
    {
      "ppmanager": get_chat_manager(global_vars.model_type),
      "model_type": global_vars.model_type
    },
    gen_config.temperature, gen_config.top_p, gen_config.top_k, gen_config.repetition_penalty, gen_config.max_new_tokens,  gen_config.num_beams, gen_config.use_cache, gen_config.do_sample, gen_config.eos_token_id, gen_config.pad_token_id,
    gen_sum_config.temperature, gen_sum_config.top_p, gen_sum_config.top_k, gen_sum_config.repetition_penalty, gen_sum_config.max_new_tokens,  gen_sum_config.num_beams, gen_sum_config.use_cache, gen_sum_config.do_sample, gen_sum_config.eos_token_id, gen_sum_config.pad_token_id,
  )


def toggle_inspector(view_selector):
    if view_selector == "with context inspector":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def reset_everything():
    return (
        [],
        {
          "ppmanager": get_chat_manager(global_vars.model_type),
          "model_type": global_vars.model_type
        },
        "",
        "",
    )

with gr.Blocks(css=MODEL_SELECTION_CSS, theme='gradio/soft') as demo:
  with gr.Column() as model_choice_view:
    gr.Markdown("# Choose a Model", elem_classes=["center"])
    with gr.Row(elem_id="container"):
      with gr.Column():
        gr.Markdown("## < 10B")
        with gr.Row():
          with gr.Column(min_width=20):
            t5_vicuna_3b = gr.Button("t5-vicuna-3b", elem_id="t5-vicuna-3b",elem_classes=["square"])
            gr.Markdown("T5 Vicuna", elem_classes=["center"])

          with gr.Column(min_width=20):
            flan3b = gr.Button("flan-3b", elem_id="flan-3b",elem_classes=["square"])
            gr.Markdown("Flan-XL", elem_classes=["center"])

          with gr.Column(min_width=20):
            camel5b = gr.Button("camel-5b", elem_id="camel-5b",elem_classes=["square"])
            gr.Markdown("Camel", elem_classes=["center"])

          with gr.Column(min_width=20):
            alpaca_lora7b = gr.Button("alpaca-lora-7b", elem_id="alpaca-lora-7b",elem_classes=["square"])
            gr.Markdown("Alpaca-LoRA", elem_classes=["center"])

          with gr.Column(min_width=20):
            stablelm7b = gr.Button("stablelm-7b", elem_id="stablelm-7b",elem_classes=["square"])
            gr.Markdown("StableLM", elem_classes=["center"])


        with gr.Row():
          with gr.Column(min_width=20):
            os_stablelm7b = gr.Button("os-stablelm-7b", elem_id="os-stablelm-7b",elem_classes=["square"])
            gr.Markdown("OS+StableLM", elem_classes=["center"])

          with gr.Column(min_width=20):
            gpt4_alpaca_7b = gr.Button("gpt4-alpaca-7b", elem_id="gpt4-alpaca-7b",elem_classes=["square"])
            gr.Markdown("GPT4-Alpaca-LoRA", elem_classes=["center"])

          with gr.Column(min_width=20):
            mpt_7b = gr.Button("mpt-7b", elem_id="mpt-7b",elem_classes=["square"])
            gr.Markdown("MPT", elem_classes=["center"])

          with gr.Column(min_width=20):
            redpajama_7b = gr.Button("redpajama-7b", elem_id="redpajama-7b",elem_classes=["square"])
            gr.Markdown("RedPajama(7B)", elem_classes=["center"])

          with gr.Column(min_width=20):
            vicuna_7b = gr.Button("vicuna-7b", elem_id="vicuna-7b",elem_classes=["square"])
            gr.Markdown("Vicuna", elem_classes=["center"])

        with gr.Row():
          with gr.Column(min_width=20):
            llama_deus_7b = gr.Button("llama-deus-7b", elem_id="llama-deus-7b",elem_classes=["square"])
            gr.Markdown("LLaMA Deus", elem_classes=["center"])

          with gr.Column(min_width=20):
            evolinstruct_vicuna_7b = gr.Button("evolinstruct-vicuna-7b", elem_id="evolinstruct-vicuna-7b",elem_classes=["square"])
            gr.Markdown("EvolInstruct Vicuna", elem_classes=["center"])

          # with gr.Column(min_width=20):
          #   _ = gr.Button("", elem_id="10b-placeholder1",elem_classes=["square"])
          #   gr.Markdown("", elem_classes=["center"])

          with gr.Column(min_width=20):
            _ = gr.Button("", elem_id="10b-placeholder2",elem_classes=["square"])
            gr.Markdown("", elem_classes=["center"])

          with gr.Column(min_width=20):
            _ = gr.Button("", elem_id="10b-placeholder3",elem_classes=["square"])
            gr.Markdown("", elem_classes=["center"])

          with gr.Column(min_width=20):
            _ = gr.Button("", elem_id="10b-placeholder4",elem_classes=["square"])
            gr.Markdown("", elem_classes=["center"])            

          # with gr.Column(min_width=20):
          #   stackllama7b = gr.Button("stackllama-7b", elem_id="stackllama-7b",elem_classes=["square"])
          #   gr.Markdown("StackLLaMA", elem_classes=["center"])            
# 
        gr.Markdown("## < 20B")
        with gr.Row():
          with gr.Column(min_width=20):
            flan11b = gr.Button("flan-11b", elem_id="flan-11b",elem_classes=["square"])
            gr.Markdown("Flan-XXL", elem_classes=["center"])

          with gr.Column(min_width=20):
            koalpaca = gr.Button("koalpaca", elem_id="koalpaca",elem_classes=["square"])
            gr.Markdown("koalpaca", elem_classes=["center"])

          with gr.Column(min_width=20):
            alpaca_lora13b = alpaca_lora13b = gr.Button("alpaca-lora-13b", elem_id="alpaca-lora-13b",elem_classes=["square"])
            gr.Markdown("Alpaca-LoRA", elem_classes=["center"])

          with gr.Column(min_width=20):
            gpt4_alpaca_13b = gr.Button("gpt4-alpaca-13b", elem_id="gpt4-alpaca-13b",elem_classes=["square"])
            gr.Markdown("GPT4-Alpaca-LoRA", elem_classes=["center"])

          with gr.Column(min_width=20):
            stable_vicuna_13b = gr.Button("stable-vicuna-13b", elem_id="stable-vicuna-13b",elem_classes=["square"])
            gr.Markdown("Stable-Vicuna", elem_classes=["center"])

        with gr.Row():
          with gr.Column(min_width=20):
            starchat_15b = gr.Button("starchat-15b", elem_id="starchat-15b",elem_classes=["square"])
            gr.Markdown("StarChat", elem_classes=["center"])

          with gr.Column(min_width=20):
            vicuna_13b = gr.Button("vicuna-13b", elem_id="vicuna-13b",elem_classes=["square"])
            gr.Markdown("Vicuna", elem_classes=["center"])

          with gr.Column(min_width=20):
            evolinstruct_vicuna_13b = gr.Button("evolinstruct-vicuna-13b", elem_id="evolinstruct-vicuna-13b",elem_classes=["square"])
            gr.Markdown("EvolInstruct Vicuna", elem_classes=["center"])

          # with gr.Column(min_width=20):
          #   gr.Button("", elem_id="20b-placeholder1",elem_classes=["square"])
          #   gr.Markdown("", elem_classes=["center"])

          # with gr.Column(min_width=20):
          #   gr.Button("", elem_id="20b-placeholder2",elem_classes=["square"])
          #   gr.Markdown("", elem_classes=["center"])

          with gr.Column(min_width=20):
            gr.Button("", elem_id="20b-placeholder3",elem_classes=["square"])
            gr.Markdown("", elem_classes=["center"])

          with gr.Column(min_width=20):
            gr.Button("", elem_id="20b-placeholder4",elem_classes=["square"])
            gr.Markdown("", elem_classes=["center"])            
          
        gr.Markdown("## < 30B")
        with gr.Row():
          with gr.Column(min_width=20):
            camel20b = gr.Button("camel-20b", elem_id="camel-20b",elem_classes=["square"])
            gr.Markdown("Camel", elem_classes=["center"])

          with gr.Column(min_width=20):
            gr.Button("", elem_id="30b-placeholder1",elem_classes=["square"])
            gr.Markdown("", elem_classes=["center"])

          with gr.Column(min_width=20):
            gr.Button("", elem_id="30b-placeholder2",elem_classes=["square"])
            gr.Markdown("", elem_classes=["center"])

          with gr.Column(min_width=20):
            gr.Button("", elem_id="30b-placeholder3",elem_classes=["square"])
            gr.Markdown("", elem_classes=["center"])

          with gr.Column(min_width=20):
            gr.Button("", elem_id="30b-placeholder4",elem_classes=["square"])
            gr.Markdown("", elem_classes=["center"])                        

        progress_view = gr.Textbox(label="Progress")

  with gr.Column(visible=False) as model_review_view:
    gr.Markdown("# Confirm the chosen model", elem_classes=["center"])
    with gr.Column(elem_id="container2"):
      with gr.Row():
        model_image = gr.Image(
            None,
            interactive=False, show_label=False
        )
        with gr.Column():
          model_name = gr.Markdown("**Model name**")
          model_params = gr.Markdown("Parameters\n: ...")
          model_base = gr.Markdown("🤗 Hub(base)\n: ...")
          model_ckpt = gr.Markdown("🤗 Hub(ckpt)\n: ...")

      with gr.Column():
        gen_config_path = gr.Dropdown(response_configs, value=response_configs[0], interactive=True, label="Gen Config(response)")
        gen_config_sum_path = gr.Dropdown(summarization_configs, value=summarization_configs[0], interactive=True, label="Gen Config(summarization)")
        with gr.Row():
          multi_gpu = gr.Checkbox(label="Multi GPU / (Non 8Bit mode)")
          chat_only_mode = gr.Checkbox(label="Chat Only Mode")

      with gr.Row():
        back_to_model_choose_btn = gr.Button("Back")
        confirm_btn = gr.Button("Confirm")

      with gr.Column():
        txt_view = gr.Textbox(label="Status")
        progress_view2 = gr.Textbox(label="Progress")
        
  with gr.Column(visible=False) as chat_view:
    chat_state = gr.State()
    
    with gr.Column(elem_id='col-container'):
        view_selector = gr.Radio(
            ["with context inspector", "without context inspector"],
            value="without context inspector",
            label="View Selector", 
            info="How do you like to use this application?",
            visible=True
        )

        with gr.Row():
            with gr.Column(elem_id='chatbot-wrapper'):
                chatbot = gr.Chatbot(elem_id='chatbot')#, label=global_vars.model_type)
                instruction_txtbox = gr.Textbox(placeholder="What do you want to say to AI?", label="Instruction")

                with gr.Row():
                    cancel_btn = gr.Button(value="Cancel")
                    reset_btn = gr.Button(value="Reset")

            with gr.Column(visible=False) as context_inspector_section:
                gr.Markdown("#### What model actually sees")
                inspector = gr.Textbox(label="", lines=28, max_lines=28, interactive=False)    

        with gr.Accordion("Constrol Panel", open=False, visible=False) as control_panel:
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

                with gr.Column():
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
                        ctx_num_lconv = gr.Slider(2, 6, 3, step=1, label="num of last talks to keep", interactive=True)
                        ctx_sum_prompt = gr.Textbox(
                            "summarize our conversations. what have we discussed about so far?",
                            label="design a prompt to summarize the conversations"
                        )

        # with gr.Accordion("Acknowledgements", open=False, visible=not args.chat_only_mode):
        #     gr.Markdown(f"{BOTTOM_LINE}")

  btns = [
    t5_vicuna_3b, flan3b, camel5b, alpaca_lora7b, stablelm7b,
    gpt4_alpaca_7b, os_stablelm7b, mpt_7b, redpajama_7b, 
    llama_deus_7b, evolinstruct_vicuna_7b,
    flan11b, koalpaca, alpaca_lora13b, gpt4_alpaca_13b, stable_vicuna_13b,
    starchat_15b, vicuna_7b, vicuna_13b, evolinstruct_vicuna_13b,
    camel20b
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
    [model_name, model_base, model_ckpt, gen_config_path, gen_config_sum_path, multi_gpu, chat_only_mode],
    [progress_view2, view_selector, control_panel]
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

  view_selector.change(
      toggle_inspector,
      view_selector,
      context_inspector_section
  )

  send_event = instruction_txtbox.submit(
      central.chat_stream,
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


demo.queue().launch(
  server_port=6006,
  server_name="0.0.0.0",
  debug=True
)