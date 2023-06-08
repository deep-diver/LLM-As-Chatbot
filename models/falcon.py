import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.padding_side = "left"

    if mode_cpu:
        print("cpu mode")
        model = AutoModelForCausalLM.from_pretrained(
            base, 
            device_map={"": "cpu"},
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
            
    elif mode_mps:
        print("mps mode")
        model = AutoModelForCausalLM.from_pretrained(
            base,
            device_map={"": "mps"},
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
            
    else:
        print("gpu mode")
        print(f"8bit = {mode_8bit}, 4bit = {mode_4bit}")
        model = AutoModelForCausalLM.from_pretrained(
            base,
            load_in_8bit=mode_8bit,
            load_in_4bit=mode_4bit,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        if not mode_8bit and not mode_4bit:
            model.half()

    # model = BetterTransformer.transform(model)
    return model, tokenizer