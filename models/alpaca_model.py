import sys
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(base, finetuned, multi_gpu, force_download_ckpt, load_8bit=False):
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:
        pass     

    model_revision = "main"
    if 'gpt-j' in base:
        model_revision = "float16"
    
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base,
            revision=model_revision,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            finetuned,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base,
            revision=model_revision,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            finetuned,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base, 
            revision=model_revision,
            device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            finetuned,
            device_map={"": device},
        )

    if not load_8bit and torch.cuda.is_available():
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return model, tokenizer