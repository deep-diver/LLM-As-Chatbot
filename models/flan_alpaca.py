import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from optimum.bettertransformer import BetterTransformer

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

def load_model(
    base, 
    finetuned, 
    gptq,
    gptq_base,
    mode_cpu,
    mode_mps,
    mode_full_gpu,
    mode_8bit,
    mode_4bit,
    mode_gptq,
    mode_mps_gptq,
    mode_cpu_gptq,
    force_download_ckpt,
    local_files_only
):
    tokenizer = AutoTokenizer.from_pretrained(
        base, local_files_only=local_files_only
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    
    if mode_cpu:
        print("cpu mode")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base, 
            device_map={"": "cpu"}, 
            low_cpu_mem_usage=True,
            local_files_only=local_files_only
        )
            
    elif mode_mps:
        print("mps mode")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base,
            device_map={"": "mps"},
            torch_dtype=torch.float16,
            local_files_only=local_files_only
        )

    elif mode_gptq:
        print("gpu(gptq) mode")
        tokenizer = AutoTokenizer.from_pretrained(
            gptq, local_files_only=local_files_only
        )
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"        
        
        model = AutoGPTQForCausalLM.from_quantized(
            gptq,
            model_basename=gptq_base,
            use_safetensors=True,
            trust_remote_code=False,
            device_map="auto",
            quantize_config=None,
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
            local_files_only=local_files_only
        )

        if not mode_8bit and not mode_4bit:
            model.half()

    model = BetterTransformer.transform(model)
    return model, tokenizer

