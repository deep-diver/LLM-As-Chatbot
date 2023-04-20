from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(base, finetuned, multi_gpu, force_download_ckpt):
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 0

    model = AutoModelForCausalLM.from_pretrained(
        base, 
        load_in_8bit=False if multi_gpu else True, 
        device_map="auto")

    model.half()
    return model, tokenizer