import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Gradio Application for LLM as a chatbot service"
    )
    parser.add_argument(
        "--base-url",
        help="Hugging Face Hub URL",
        default="elinas/llama-7b-hf-transformers-4.29",
        type=str,
    )
    parser.add_argument(
        "--ft-ckpt-url",
        help="Hugging Face Hub URL",
        # default="tloen/alpaca-lora-7b",
        default="LLMs/Alpaca-LoRA-7B-elina",
        type=str,
    )
    parser.add_argument(
        "--port",
        help="PORT number where the app is served",
        default=6006,
        type=int,
    )
    parser.add_argument(
        "--share",
        help="Create and share temporary endpoint (useful in Colab env)",
        action='store_true'
    )
    parser.add_argument(
        "--gen-config-path",
        help="path to GenerationConfig file",
        default="configs/response_configs/default.yaml",
        # default="configs/gen_config_koalpaca.yaml",
        # default="configs/gen_config_stablelm.yaml",
        type=str
    )
    parser.add_argument(
        "--gen-config-summarization-path",
        help="path to GenerationConfig file used in context summarization",
        default="configs/summarization_configs/default.yaml",
        type=str
    )
    parser.add_argument(
        "--multi-gpu",
        help="Enable multi gpu mode. This will force not to use Int8 but float16, so you need to check if your system has enough GPU memory",
        action='store_true'
    )
    parser.add_argument(
        "--force-download_ckpt",
        help="Force to download ckpt instead of using cached one",
        action="store_true"
    )
    parser.add_argument(
        "--chat-only-mode",
        help="Only show chatting window. Otherwise, other components will be appeared for more sophisticated control",
        action="store_true"
    )
    
    return parser.parse_args()