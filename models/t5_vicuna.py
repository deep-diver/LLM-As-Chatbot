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
    force_download_ckpt,
    local_files_only
):
    tokenizer = T5Tokenizer.from_pretrained(
        base, use_fast=False, local_files_only=local_files_only
    )
    tokenizer.padding_side = "left"
    
    if mode_cpu:
        print("cpu mode")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base, 
            device_map={"": "cpu"}, 
            use_safetensors=False,
            local_files_only=local_files_only
        )
            
    elif mode_mps:
        print("mps mode")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base,
            device_map={"": "mps"},
            torch_dtype=torch.float16,
            use_safetensors=False,
            local_files_only=local_files_only
        )
            
    else:
        print("gpu mode")
        print(f"8bit = {mode_8bit}, 4bit = {mode_4bit}")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base,
            load_in_8bit=mode_8bit,
            load_in_4bit=mode_4bit,
            device_map="auto",
            torch_dtype=torch.float16,
            use_safetensors=False,
            local_files_only=local_files_only
        )

        if not mode_8bit and not mode_4bit:
            model.half()

    model = BetterTransformer.transform(model)
    return model, tokenizer