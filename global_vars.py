import yaml
from transformers import GenerationConfig
from models import alpaca, stablelm, koalpaca, flan_alpaca, mpt
from models import camel, t5_vicuna, vicuna, starchat, redpajama, bloom
from models import baize, guanaco

def initialize_globals(args):
    global model, model_type, stream_model, tokenizer
    global gen_config, gen_config_raw    
    global gen_config_summarization
    
    model_type_tmp = "alpaca"
    if "baize" in args.base_url.lower():
        model_type_tmp = "baize"
    elif "vicuna" in args.base_url.lower():
        model_type_tmp = "vicuna"
    elif "mpt" in args.base_url.lower():
        model_type_tmp = "mpt"
    elif "redpajama" in args.base_url.lower():
        model_type_tmp = "redpajama"
    elif "starchat" in args.base_url.lower():
        model_type_tmp = "starchat"
    elif "camel" in args.base_url.lower():
        model_type_tmp = "camel"
    elif "flan-alpaca" in args.base_url.lower():
        model_type_tmp = "flan-alpaca"
    elif "openassistant/stablelm" in args.base_url.lower():
        model_type_tmp = "os-stablelm"
    elif "stablelm" in args.base_url.lower():
        model_type_tmp = "stablelm"
    elif "fastchat-t5" in args.base_url.lower():
        model_type_tmp = "t5-vicuna"
    elif "koalpaca-polyglot" in args.base_url.lower():
        model_type_tmp = "koalpaca-polyglot"
    elif "stable-vicuna" in args.base_url.lower():
        model_type_tmp = "stable-vicuna"
    elif "alpacagpt4" in args.ft_ckpt_url.lower():
        model_type_tmp = "alpaca-gpt4"
    elif "alpaca" in args.ft_ckpt_url.lower():
        model_type_tmp = "alpaca"
    elif "llama-deus" in args.ft_ckpt_url.lower():
        model_type_tmp = "llama-deus"
    elif "vicuna-lora-evolinstruct" in args.ft_ckpt_url.lower():
        model_type_tmp = "evolinstruct-vicuna"
    elif "alpacoom" in args.ft_ckpt_url.lower():
        model_type_tmp = "alpacoom"
    elif "guanaco" in args.ft_ckpt_url.lower():
        model_type_tmp = "guanaco"
    else:
        print("unsupported model type")
        quit()

    print(f"determined model type: {model_type_tmp}")        

    try:
        if model is not None:
            del model

        if tokenizer is not None:
            del tokenizer
    except NameError:
        pass

    load_model = get_load_model(model_type_tmp)
    model, tokenizer = load_model(
        base=args.base_url,
        finetuned=args.ft_ckpt_url,
        multi_gpu=args.multi_gpu,
        force_download_ckpt=args.force_download_ckpt
    )        
        
    gen_config, gen_config_raw = get_generation_config(args.gen_config_path)
    gen_config_summarization, _ = get_generation_config(args.gen_config_summarization_path)
    model_type = model_type_tmp
    stream_model = model
        
def get_load_model(model_type):
    if model_type == "alpaca" or \
        model_type == "alpaca-gpt4" or \
        model_type == "llama-deus":
        return alpaca.load_model
    elif model_type == "stablelm" or model_type == "os-stablelm":
        return stablelm.load_model
    elif model_type == "koalpaca-polyglot":
        return koalpaca.load_model
    elif model_type == "flan-alpaca":
        return flan_alpaca.load_model
    elif model_type == "camel":
        return camel.load_model
    elif model_type == "t5-vicuna":
        return t5_vicuna.load_model
    elif model_type == "stable-vicuna":
        return vicuna.load_model
    elif model_type == "starchat":
        return starchat.load_model
    elif model_type == "mpt":
        return mpt.load_model
    elif model_type == "redpajama":
        return redpajama.load_model
    elif model_type == "vicuna":
        return vicuna.load_model
    elif model_type == "evolinstruct-vicuna":
        return alpaca.load_model
    elif model_type == "alpacoom":
        return bloom.load_model
    elif model_type == "baize":
        return baize.load_model
    elif model_type == "guanaco":
        return guanaco.load_model
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
