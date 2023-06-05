import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.bettertransformer import BetterTransformer

def load_model(base, finetuned, multi_gpu, force_download_ckpt):
    tokenizer = AutoTokenizer.from_pretrained(base)

    model = AutoModelForCausalLM.from_pretrained(
        base,
        load_in_8bit=False if multi_gpu else True,
        device_map="auto",
    )

    if multi_gpu:
        model.half()
    
    if finetuned is not None and \
        finetuned != "" and \
        finetuned != "N/A":

        model = PeftModel.from_pretrained(
            model, 
            finetuned, 
            # force_download=force_download_ckpt,
        )
       
    # model = BetterTransformer.transform(model)
    return model, tokenizer