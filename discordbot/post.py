def clean(text):
    if text.endswith("</s>"):
        text = text[:-len("</s>")]

    if text.endswith("<|endoftext|>"):
        text = text[:-len("<|endoftext|>")]
        
    return text