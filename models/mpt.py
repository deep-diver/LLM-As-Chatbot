import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from optimum.bettertransformer import BetterTransformer

def load_model(base, finetuned, multi_gpu, force_download_ckpt):
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    tokenizer.padding_side = "left"

    config = AutoConfig.from_pretrained(
        base,
        trust_remote_code=True
    )
    config.attn_config['attn_impl'] = 'triton'
    model = AutoModelForCausalLM.from_pretrained(
        base, 
        # config=config,
        # load_in_8bit=False if multi_gpu else True, 
        # device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).cuda()

    if multi_gpu:
        model.half()

    # model = BetterTransformer.transform(model)
    return model, tokenizer