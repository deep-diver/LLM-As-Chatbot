from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM
from utils import get_device

device = get_device()

def load_model(
    base="decapoda-research/llama-7b-hf",
    finetuned="tloen/alpaca-lora-7b",
):
    tokenizer = LlamaTokenizer.from_pretrained(base)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model, finetuned,
            torch_dtype=torch.float16
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base,
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
        model = LlamaForCausalLM.from_pretrained(
            base,
            device_map={"": device},
            low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            finetuned,
            device_map={"": device},
        )

    return model, tokenizer

