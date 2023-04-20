import yaml
from transformers import GenerationConfig
from models import alpaca, stablelm

def initialize_globals(args):
    global model, model_type, stream_model, tokenizer
    global gen_config, gen_config_raw    
    global gen_config_summarization
    
    model_type = "alpaca"

    if "stablelm" in args.base_url:
        model_type = "stablelm"
    elif "gpt4-alpaca" in args.ft_ckpt_url:
        model_type = "alpaca-gpt4"
    elif "alpaca" in args.ft_ckpt_url:
        model_type = "alpaca"
    else:
        print("unsupported model type")
        quit()

    print(f"determined model type: {model_type}")        
    load_model = get_load_model(model_type)
    model, tokenizer = load_model(
        base=args.base_url,
        finetuned=args.ft_ckpt_url,
        multi_gpu=args.multi_gpu,
        force_download_ckpt=args.force_download_ckpt
    )        
        
    gen_config, gen_config_raw = get_generation_config(args.gen_config_path)
    gen_config_summarization, _ = get_generation_config(args.gen_config_summarization_path)
    
    if not batch_enabled:
        stream_model = model
        
def get_load_model(model_type):
    if model_type == "alpaca" or model_type == "alpaca-gpt4":
        return alpaca.load_model
    elif model_type == "stablelm":
        return stablelm.load_model
    else:
        return None
    
def get_generation_config(path):
    with open(path, 'rb') as f:
        generation_config = yaml.safe_load(f.read())
        
    generation_config = generation_config["generation_config"]

    return GenerationConfig(**generation_config), generation_config

def get_constraints_config(path):
    with open(path, 'rb') as f:
        constraints_config = yaml.safe_load(f.read())
        
    return ConstraintsConfig(**constraints_config), constraints_config["constraints"]
