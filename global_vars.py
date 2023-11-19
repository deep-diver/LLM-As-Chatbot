import gc
import yaml
import torch
from transformers import GenerationConfig
from models import alpaca, stablelm, koalpaca, flan_alpaca, mpt
from models import camel, t5_vicuna, vicuna, starchat, redpajama, bloom
from models import baize, guanaco, falcon, kullm, replit, airoboros
from models import samantha_vicuna, wizard_coder, xgen, freewilly
from models import mistral
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
    global model_thumbnail_tiny, device
    global gen_config, gen_config_raw
    global gen_config_summarization

    model_type = "custom"

    model, tokenizer = byom.load_model(
        base=base,
        finetuned=ckpt,
        mode_cpu=mode_cpu,
        mode_mps=mode_mps,
        mode_full_gpu=mode_full_gpu,
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
    global device, model_thumbnail_tiny, model_name
    global model, model_type, stream_model, tokenizer
    global remote_addr, remote_port, remote_token
    global gen_config, gen_config_raw    
    global gen_config_summarization
    
    model_type_tmp = "alpaca"
    print(args.base_url.lower())
    if "mistralai/mistral" in args.base_url.lower():
        model_type_tmp = "mistral"
    elif "huggingfaceh4/zephyr" in args.base_url.lower():
        model_type_tmp = "zephyr"
    elif "meta-llama/llama-2-70b-hf" in args.base_url.lower():
        model_type_tmp = "llama2-70b"
    elif "codellama/codellama-34b-instruct-hf" in args.base_url.lower():
        model_type_tmp = "codellama2-70b"
    elif "nousresearch/nous-hermes-llama2-70b" in args.base_url.lower():
        model_type_tmp = "nous-hermes2"
    elif "mayaph/godzilla2-70b" in args.base_url.lower():
        model_type_tmp = "godzilla2"
    elif "ehartford/samantha-1.11-70b" in args.base_url.lower():
        model_type_tmp = "samantha2"
    elif "psmathur/orca_mini_v3_70b" in args.base_url.lower():
        model_type_tmp = "orcamini2"
    elif "wizardlm/wizardlm-70b" in args.base_url.lower():
        model_type_tmp = "wizardlm2"
    elif "garage-baind/platypus2-70b" in args.base_url.lower():
        model_type_tmp = "platypus2"
    elif "stable-beluga2-70b" in args.base_url.lower():
        model_type_tmp = "stable-beluga2"
    elif "redmond-puffin-" in args.base_url.lower():
        model_type_tmp = "puffin"
    elif "upstage/llama-2-70b" in args.base_url.lower():
        model_type_tmp = "upstage-llama2"
    elif "upstage/llama-" in args.base_url.lower():
        model_type_tmp = "upstage-llama"
    elif "codellama/codellama-" in args.base_url.lower():
        model_type_tmp = "codellama"        
    elif "llama-2" in args.base_url.lower():
        model_type_tmp = "llama2"
    elif "xgen" in args.base_url.lower():
        model_type_tmp = "xgen"
    elif "orca_mini" in args.base_url.lower():
        model_type_tmp = "orcamini"
    elif "open-llama" in args.base_url.lower():
        model_type_tmp = "openllama"
    elif "wizardcoder" in args.base_url.lower():
        model_type_tmp = "wizard-coder"
    elif "wizard-vicuna" in args.base_url.lower():
        model_type_tmp = "wizard-vicuna"
    elif "llms/wizardlm" in args.base_url.lower() or \
        "wizardlm/wizardlm" in args.base_url.lower():
        model_type_tmp = "wizardlm"
    elif "chronos" in args.base_url.lower():
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
    if args.mode_remote_tgi:
        device = "cpu"
    elif args.mode_cpu or args.mode_cpu_gptq:
        device = "cpu"
    elif args.mode_mps or args.mode_mps_gptq:
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
    model_name = args.model_name
    remote_addr = None
    remote_port = None
    remote_token = None
    
    if not args.mode_remote_tgi:
        load_model = get_load_model(model_type_tmp)
        model, tokenizer = load_model(
            base=args.base_url,
            finetuned=args.ft_ckpt_url,
            gptq=args.gptq_url,
            gptq_base=args.gptq_base_url,
            mode_cpu=args.mode_cpu,
            mode_mps=args.mode_mps,
            mode_full_gpu=args.mode_full_gpu,
            mode_8bit=args.mode_8bit,
            mode_4bit=args.mode_4bit,
            mode_gptq=args.mode_gptq,
            mode_mps_gptq=args.mode_mps_gptq,
            mode_cpu_gptq=args.mode_cpu_gptq,
            force_download_ckpt=args.force_download_ckpt,
            local_files_only=args.local_files_only
        )
        model.eval()
        stream_model = model
    else:
        remote_addr = args.remote_addr
        remote_port = args.remote_port
        remote_token = args.remote_token
    
    model_thumbnail_tiny = args.thumbnail_tiny
    gen_config, gen_config_raw = get_generation_config(args.gen_config_path)
    gen_config_summarization, _ = get_generation_config(args.gen_config_summarization_path)
        
def get_load_model(model_type):
    if model_type == "alpaca" or \
        model_type == "alpaca-gpt4" or \
        model_type == "llama-deus" or \
        model_type == "nous-hermes" or \
        model_type == "lazarus" or \
        model_type == "chronos" or \
        model_type == "wizardlm" or \
        model_type == "openllama" or \
        model_type == "orcamini" or \
        model_type == "llama2" or \
        model_type == "upstage-llama" or \
        model_type == "puffin" or \
        model_type == "codellama":
        return alpaca.load_model
    elif model_type == "stable-beluga2" or \
        model_type == "upstage-llama2" or \
        model_type == "platypus2" or \
        model_type == "wizardlm2" or \
        model_type == "orcamini2" or \
        model_type == "samantha2" or \
        model_type == "godzilla2" or \
        model_type == "nous-hermes2" or \
        model_type == "llama2-70b" or \
        model_type == "codellama2-70b":
        return freewilly.load_model
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
        return alpaca.load_model
    elif model_type == "starchat":
        return starchat.load_model
    elif model_type == "wizard-coder":
        return wizard_coder.load_model
    elif model_type == "mpt":
        return mpt.load_model
    elif model_type == "redpajama" or \
        model_type == "redpajama-instruct":
        return redpajama.load_model
    elif model_type == "vicuna":
        return alpaca.load_model
    elif model_type == "evolinstruct-vicuna" or \
        model_type == "wizard-vicuna":
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
    elif model_type == "xgen":
        return xgen.load_model
    elif model_type == "mistral":
        return mistral.load_model
    elif model_type == "zephyr":
        return mistral.load_model
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
