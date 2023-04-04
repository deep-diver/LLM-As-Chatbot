import torch
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM

def load_model(base, finetuned, multi_gpu):
    tokenizer = LlamaTokenizer.from_pretrained(base)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    if not multi_gpu:
        model = LlamaForCausalLM.from_pretrained(
            base,
            load_in_8bit=True,
            device_map="auto",
        )
        
        model = PeftModel.from_pretrained(
            model, finetuned, device_map={'': 0}
        )
        return model, tokenizer
    else:
        model = LlamaForCausalLM.from_pretrained(
            base,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        model = PeftModel.from_pretrained(
            model, finetuned, torch_dtype=torch.float16
        )
        model.half()
        return model, tokenizer        

