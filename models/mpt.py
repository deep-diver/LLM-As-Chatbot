import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from optimum.bettertransformer import BetterTransformer

def load_model(
    base, 
    finetuned, 
    mode_cpu,
    mode_mps,
    mode_full_gpu,
    mode_8bit,
    mode_4bit,
    force_download_ckpt
):
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    tokenizer.padding_side = "left"

    config = AutoConfig.from_pretrained(
        base,
        trust_remote_code=True
    )
    # config.attn_config['attn_impl'] = 'triton'
    model = AutoModelForCausalLM.from_pretrained(
        base, 
        # config=config,
        load_in_8bit=mode_8bit,
        load_in_4bit=mode_4bit,
        # device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).cuda()

    if not mode_8bit and not mode_4bit:
        model.half()

    # model = BetterTransformer.transform(model)
    return model, tokenizer
