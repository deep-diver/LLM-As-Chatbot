import torch

def get_output_batch(
    model, tokenizer, prompts, generation_config
):
    if len(prompts) == 1:
        encoding = tokenizer(prompts, return_tensors="pt")
        input_ids = encoding["input_ids"].cuda()
        generated_id = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            max_new_tokens=256
        )

        decoded = tokenizer.batch_decode(generated_id)
        del input_ids, generated_id
        torch.cuda.empty_cache()
        return decoded
    else:
        encodings = tokenizer(prompts, padding=True, return_tensors="pt").to('cuda')
        generated_ids = model.generate(
            **encodings,
            generation_config=generation_config,
            max_new_tokens=256
        )

        decoded = tokenizer.batch_decode(generated_ids)
        del encodings, generated_ids
        torch.cuda.empty_cache()
        return decoded
