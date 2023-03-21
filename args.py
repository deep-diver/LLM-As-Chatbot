import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Gradio Application for Alpaca-LoRA as a chatbot service"
    )
    # Dataset related.
    parser.add_argument(
        "--base_url",
        help="huggingface hub url",
        default="decapoda-research/llama-7b-hf",
        type=str,
    )
    parser.add_argument(
        "--ft_ckpt_url",
        help="huggingface hub url",
        default="tloen/alpaca-lora-7b",
        type=str,
    )
    parser.add_argument(
        "--port",
        help="port to serve app",
        default=6006,
        type=int,
    )
    parser.add_argument(
        "--batch_size",
        help="how many requests to handle at the same time",
        default=1,
        type=int
    )        
    parser.add_argument(
        "--api_open",
        help="do you want to open as API",
        default="no",
        type=str,
    )
    parser.add_argument(
        "--share",
        help="do you want to share temporarily",
        default="no",
        type=str
    )
    parser.add_argument(
        "--gen_config_path",
        help="which config to use for GenerationConfig",
        default="generation_config_default.yaml",
        type=str
    )

    return parser.parse_args()