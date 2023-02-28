from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def get_pretrained_models(
  model_path="google/flan-t5-small", 
  max_length=10000, 
  min_length=100):
  
  model = AutoModelForSeq2SeqLM.from_pretrained(
      model_path, 
      min_length=min_length, 
      max_length=max_length,
      use_cache=False,
      temperature=2.0,
      do_sample=True)
  tokenizer = AutoTokenizer.from_pretrained(model_path)
  
  return model, tokenizer

def get_output(model, tokenizer, prompt):
  inputs = tokenizer(prompt, return_tensors="pt")
  outputs = model.generate(**inputs)

  return tokenizer.batch_decode(outputs, skip_special_tokens=True)
