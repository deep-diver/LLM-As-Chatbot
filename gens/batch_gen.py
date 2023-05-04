import torch

def get_output_batch(
    model, tokenizer, prompts, generation_config, device='cuda'
):
    if len(prompts) == 1:
        encoding = tokenizer(prompts, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        generated_id = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
        )

        decoded = tokenizer.batch_decode(
            generated_id, skip_prompt=True, skip_special_tokens=True
        )
        del input_ids, generated_id
        torch.cuda.empty_cache()
        return decoded
    else:
        encodings = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        generated_ids = model.generate(
            **encodings,
            generation_config=generation_config,
        )

        decoded = tokenizer.batch_decode(
            generated_ids, skip_prompt=True, skip_special_tokens=True
        )
        del encodings, generated_ids
        torch.cuda.empty_cache()
        return decoded
