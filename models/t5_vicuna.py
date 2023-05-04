import torch
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer

def load_model(base, finetuned, multi_gpu, force_download_ckpt):
    tokenizer = T5Tokenizer.from_pretrained(base, use_fast=False)
    tokenizer.padding_side = "left"

    model = AutoModelForSeq2SeqLM.from_pretrained(
        base, 
        torch_dtype=torch.float16,
        load_in_8bit=False if multi_gpu else True, 
        device_map="auto")

    if multi_gpu: 
        model.half()
    return model, tokenizer