import torch
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM

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
    tokenizer = LlamaTokenizer.from_pretrained(
        base,local_files_only=local_files_only
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    if not multi_gpu:
        model = LlamaForCausalLM.from_pretrained(
            base,
            load_in_8bit=mode_8bit,
            load_in_4bit=mode_4bit,
            device_map="auto",
            local_files_only=local_files_only
        )
        
        model = PeftModel.from_pretrained(
            model, 
            finetuned, 
            # force_download=force_download_ckpt,
            device_map={'': 0}
        )
        return model, tokenizer
    else:
        model = LlamaForCausalLM.from_pretrained(
            base,
            load_in_8bit=mode_8bit,
            load_in_4bit=mode_4bit,            
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=local_files_only
        )
        
        model = PeftModel.from_pretrained(
            model, 
            finetuned, 
            # force_download=force_download_ckpt,
            torch_dtype=torch.float16
        )
        model.half()
        return model, tokenizer        

