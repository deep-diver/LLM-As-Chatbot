import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model(base, finetuned, multi_gpu, force_download_ckpt):

    model = AutoModelForSeq2SeqLM.from_pretrained(
        base, 
        load_in_8bit=False if multi_gpu else True, 
        device_map="auto")
    
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    
    return model, tokenizer

