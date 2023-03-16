from peft import PeftModel
from transformers import LLaMATokenizer, LLaMAForCausalLM

def load_model(
    base="decapoda-research/llama-7b-hf",
    finetuned="tloen/alpaca-lora-7b"
):
    tokenizer = LLaMATokenizer.from_pretrained(base)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    model = LLaMAForCausalLM.from_pretrained(
        base,
        load_in_8bit=True,
        device_map="auto",
    )
    
    model = PeftModel.from_pretrained(model, finetuned)
    return model, tokenizer
