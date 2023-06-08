import torch
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
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
    tokenizer = T5Tokenizer.from_pretrained(base, use_fast=False)
    tokenizer.padding_side = "left"

    model = AutoModelForSeq2SeqLM.from_pretrained(
        base, 
        torch_dtype=torch.float16,
        load_in_8bit=mode_8bit, 
        load_in_4bit=mode_4bit,
        device_map="auto")

    model = BetterTransformer.transform(model)
    return model, tokenizer