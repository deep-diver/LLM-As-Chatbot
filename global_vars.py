from models import alpaca_model
from models import flan_alpaca
from models import llama_rlhf
from gens.stream_gen import StreamModel

from miscs.utils import get_generation_config, get_constraints_config

def initialize_globals(args):
    global model, stream_model, tokenizer
    global gen_config_raw, gen_config_summarization_raw, constraints_config_raw
    global gen_config, gen_config_summarization, constraints_config
    global model_type, batch_enabled
    
    model_type = "alpaca"
    batch_enabled = True if args.batch_size > 1 else False    

    if "flan" in args.base_url:
        model_type = "flan"
    elif "alpaca" in args.ft_ckpt_url:
        model_type = "alpaca"
    elif "baize" in args.ft_ckpt_url:
        model_type = "baize"
    elif "llama" in args.ft_ckpt_url:
        model_type = "llama"
    else:
        print("unsupported model type. only llamastack, alpaca, flan, and baize are supported")
        quit()

    load_model = get_load_model(model_type)
    model, tokenizer = load_model(
        base=args.base_url,
        finetuned=args.ft_ckpt_url,
        multi_gpu=args.multi_gpu,
        force_download_ckpt=args.force_download_ckpt
    )        
        
    gen_config, gen_config_raw = get_generation_config(args.gen_config_path)
    gen_config_summarization, gen_config_summarization_raw = get_generation_config(args.gen_config_summarization_path)
    constraints_config, constraints_config_raw = get_constraints_config(args.get_constraints_config_path)
    
    if not batch_enabled:
        if model_type == 'alpaca':
            stream_model = StreamModel(model, tokenizer)
        else:
            stream_model = model
        
def get_load_model(model_type):
    if model_type == "alpaca":
        return alpaca_model.load_model
    elif model_type == "flan":
        return flan_alpaca.load_model
    elif model_type == "llama":
        return llama_rlhf.load_model
    else:
        return None    