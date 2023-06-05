import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.bettertransformer import BetterTransformer

def load_model(base, finetuned, multi_gpu, force_download_ckpt):
    tokenizer = AutoTokenizer.from_pretrained(
        base, trust_remote_code=True)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base,
        load_in_8bit=False if multi_gpu else True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
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

