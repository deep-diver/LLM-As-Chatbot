from models.alpaca_model import load_model
from gens.stream_gen import StreamModel

from miscs.utils import get_generation_config

def initialize_globals(args):
    global model, stream_model, tokenizer
    global generation_config, gen_config_summarization
    global model_type, batch_enabled
    
    model_type = "alpaca"
    batch_enabled = True if args.batch_size > 1 else False    

    model, tokenizer = load_model(
        base=args.base_url,
        finetuned=args.ft_ckpt_url,
        multi_gpu=args.multi_gpu,
        force_download_ckpt=args.force_download_ckpt
    )
    
    if "alpaca" in args.ft_ckpt_url:
        model_type = "alpaca"
    elif "baize" in args.ft_ckpt_url:
        model_type = "baize"
    else:
        print("unsupported model type. only alpaca and baize are supported")
        quit()
    
    generation_config = get_generation_config(
        args.gen_config_path
    )
    gen_config_summarization = get_generation_config(
        args.gen_config_summarization_path
    )
    
    if not batch_enabled:
        stream_model = StreamModel(model, tokenizer)    