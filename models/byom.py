import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(
    base, 
    finetuned, 
    multi_gpu, 
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

    model = model_cls.from_pretrained(
        base,
        load_in_8bit=False if multi_gpu else True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if finetuned is not None and \
        finetuned != "" and \
        finetuned != "N/A":
        model = PeftModel.from_pretrained(
            model, 
            finetuned, 
            force_download=force_download_ckpt,
            device_map={'': 0}
        )
    
    return model, tokenizer