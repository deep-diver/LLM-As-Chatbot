import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(
    base, 
    finetuned, 
    mode_cpu=mode_cpu,
    mode_mps=mode_mps,
    mode_full_gpu=mode_full_gpu,
    mode_8bit=mode_8bit,
    mode_4bit=mode_4bit,
    force_download_ckpt,
    model_cls,
    tokenizer_cls
):
    if tokenizer_cls is None:
        tokenizer_cls = AutoTokenizer
    else:
        tokenizer_cls = eval(tokenizer_cls)
    
    if model_cls is None:
        model_cls = AutoModelForCausalLM
    else:
        model_cls = eval(model_cls)

    print(f"tokenizer_cls: {tokenizer_cls}")
    print(f"model_cls: {model_cls}")
    
    tokenizer = tokenizer_cls.from_pretrained(base)
    tokenizer.padding_side = "left"

    if mode_cpu:
        print("cpu mode")
        model = model_cls.from_pretrained(
            base, 
            device_map={"": "cpu"}, 
            use_safetensors=False
            # low_cpu_mem_usage=True
        )
        
        if finetuned is not None and \
            finetuned != "" and \
            finetuned != "N/A":
            model = PeftModel.from_pretrained(
                model, 
                finetuned,
                device_map={"": "cpu"}
                # force_download=force_download_ckpt,
            )
    elif mode_mps:
        print("mps mode")
        model = model_cls.from_pretrained(
            base,
            device_map={"": "mps"},
            torch_dtype=torch.float16,
            use_safetensors=False
        )
        
        if finetuned is not None and \
            finetuned != "" and \
            finetuned != "N/A":
            model = PeftModel.from_pretrained(
                model, 
                finetuned,
                torch_dtype=torch.float16,
                device_map={"": "mps"}
                # force_download=force_download_ckpt,
            )
    else:
        print("gpu mode")
        print(f"8bit = {mode_8bit}, 4bit = {mode_4bit}")
    
        model = model_cls.from_pretrained(
            base,
            load_in_8bit=mode_8bit,
            load_in_4bit=mode_4bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        if finetuned is not None and \
            finetuned != "" and \
            finetuned != "N/A":
            model = PeftModel.from_pretrained(
                model, 
                finetuned, 
                # force_download=force_download_ckpt,
            )

        return model, tokenizer