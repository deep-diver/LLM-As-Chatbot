import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.bettertransformer import BetterTransformer

def load_model(base, finetuned, multi_gpu, force_download_ckpt):
    tokenizer = AutoTokenizer.from_pretrained(base)

    if not multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(
            base,
            load_in_8bit=True,
            device_map="auto",
        )
        
        model = PeftModel.from_pretrained(
            model, 
            finetuned, 
            # force_download=force_download_ckpt,
            device_map={'': 0}
        )
        # model = BetterTransformer.transform(model)
        return model, tokenizer
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        model = PeftModel.from_pretrained(
            model, 
            finetuned, 
            # force_download=force_download_ckpt,
            torch_dtype=torch.float16
        )
        model.half()
        # model = BetterTransformer.transform(model)
        return model, tokenizer