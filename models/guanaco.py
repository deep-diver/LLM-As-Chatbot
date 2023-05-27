import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer


def load_model(base, finetuned, multi_gpu, force_download_ckpt):
    tokenizer = LlamaTokenizer.from_pretrained(base)
    tokenizer.bos_token_id = 1
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base,
        load_in_8bit=False if multi_gpu else True,
        torch_dtype=torch.bfloat16,
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
    # model = BetterTransformer.transform(model)
    return model, tokenizer