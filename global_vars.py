import gc
import yaml
import torch
from transformers import GenerationConfig
from models import alpaca, stablelm, koalpaca, flan_alpaca, mpt
from models import camel, t5_vicuna, vicuna, starchat, redpajama, bloom
from models import baize, guanaco, falcon, kullm, replit, airoboros
from models import samantha_vicuna
from models import byom

cuda_availability = False
available_vrams_gb = 0
mps_availability = False

if torch.cuda.is_available():
    cuda_availability = True
    available_vrams_mb = sum(
        [
            torch.cuda.get_device_properties(i).total_memory 
            for i in range(torch.cuda.device_count())
        ]
    ) / 1024. / 1024
    
if torch.backends.mps.is_available():
    mps_availability = True
 
def initialize_globals_byom(
    base, ckpt, model_cls, tokenizer_cls, 
    bos_token_id, eos_token_id, pad_token_id,
    mode_cpu, model_mps, mode_8bit, mode_4bit, mode_full_gpu
):
    global model, model_type, stream_model, tokenizer
    global gen_config, gen_config_raw
    global gen_config_summarization

    model_type = "custom"

    model, tokenizer = byom.load_model(
        base=base,
        finetuned=ckpt,
        mode_8bit=mode_8bit,
        mode_4bit=mode_4bit,
        model_cls=model_cls if model_cls != "" else None,
        tokenizer_cls=tokenizer_cls if tokenizer_cls != "" else None
    )
    
    stream_model = model
    gen_config, gen_config_raw = get_generation_config("configs/response_configs/default.yaml")
    gen_config_summarization, _ = get_generation_config("configs/summarization_configs/default.yaml")
    if bos_token_id != "" or bos_token_id.isdigit():
        gen_config.bos_token_id = int(bos_token_id)

    if eos_token_id != "" or eos_token_id.isdigit():
        gen_config.eos_token_id = int(eos_token_id)

    if pad_token_id != "" or pad_token_id.isdigit():
        gen_config.pad_token_id = int(pad_token_id)       

def initialize_globals(args):
    global device
    global model, model_type, stream_model, tokenizer
    global gen_config, gen_config_raw    
    global gen_config_summarization
    
    model_type_tmp = "alpaca"
    if "chronos" in args.base_url.lower():
        model_type_tmp = "chronos"
    elif "lazarus" in args.base_url.lower():
        model_type_tmp = "lazarus"
    elif "samantha" in args.base_url.lower():
        model_type_tmp = "samantha-vicuna"
    elif "airoboros" in args.base_url.lower():
        model_type_tmp = "airoboros"
    elif "replit" in args.base_url.lower():
        model_type_tmp = "replit-instruct"
    elif "kullm" in args.base_url.lower():
        model_type_tmp = "kullm-polyglot"
    elif "nous-hermes" in args.base_url.lower():
        model_type_tmp = "nous-hermes"
    elif "guanaco" in args.base_url.lower():
        model_type_tmp = "guanaco"
    elif "wizardlm-uncensored-falcon" in args.base_url.lower():
        model_type_tmp = "wizard-falcon"        
    elif "falcon" in args.base_url.lower():
        model_type_tmp = "falcon"
    elif "baize" in args.base_url.lower():
        model_type_tmp = "baize"
    elif "stable-vicuna" in args.base_url.lower():
        model_type_tmp = "stable-vicuna"        
    elif "vicuna" in args.base_url.lower():
        model_type_tmp = "vicuna"
    elif "mpt" in args.base_url.lower():
        model_type_tmp = "mpt"
    elif "redpajama-incite-7b-instruct" in args.base_url.lower():
        model_type_tmp = "redpajama-instruct"
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

    device = "cpu"
    if args.mode_cpu:
        device = "cpu"
    elif args.mode_mps:
        device = "mps"
    else:
        device = "cuda"
    
    try:
        if model is not None:
            del model

        if stream_model is not None:
            del stream_model

        if tokenizer is not None:
            del tokenizer

        gc.collect()
        
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
            
    except NameError:
        pass
        
    model_type = model_type_tmp
    load_model = get_load_model(model_type_tmp)
    model, tokenizer = load_model(
        base=args.base_url,
        finetuned=args.ft_ckpt_url,
        mode_cpu=args.mode_cpu,
        mode_mps=args.mode_mps,
        mode_full_gpu=args.mode_full_gpu,
        mode_8bit=args.mode_8bit,
        mode_4bit=args.mode_4bit,
        force_download_ckpt=args.force_download_ckpt
    )
        
    gen_config, gen_config_raw = get_generation_config(args.gen_config_path)
    gen_config_summarization, _ = get_generation_config(args.gen_config_summarization_path)
    stream_model = model
        
def get_load_model(model_type):
    if model_type == "alpaca" or \
        model_type == "alpaca-gpt4" or \
        model_type == "llama-deus" or \
        model_type == "nous-hermes" or \
        model_type == "lazarus" or \
        model_type == "chronos":
        return alpaca.load_model
    elif model_type == "stablelm" or model_type == "os-stablelm":
        return stablelm.load_model
    elif model_type == "koalpaca-polyglot":
        return koalpaca.load_model
    elif model_type == "kullm-polyglot":
        return kullm.load_model
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
    elif model_type == "redpajama" or \
        model_type == "redpajama-instruct":
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
    elif model_type == "falcon" or model_type == "wizard-falcon":
        return falcon.load_model
    elif model_type == "replit-instruct":
        return replit.load_model
    elif model_type == "airoboros":
        return airoboros.load_model
    elif model_type == "samantha-vicuna":
        return samantha_vicuna.load_model
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
