import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
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
        base, trust_remote_code=True, local_files_only=local_files_only
    )
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base,
        load_in_8bit=mode_8bit,
        load_in_4bit=mode_4bit,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=local_files_only
    )

    if finetuned is not None and \
        finetuned != "" and \
        finetuned != "N/A":

        model = PeftModel.from_pretrained(
            model, 
            finetuned, 
            # force_download=force_download_ckpt,
            trust_remote_code=True
        )

        model = model.merge_and_unload()

    # model = BetterTransformer.transform(model)
    model.to('cuda')
    return model, tokenizer

