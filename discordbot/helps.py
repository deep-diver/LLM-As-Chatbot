def get_help():
    help_msg = """Type one of the following for more information about this chatbot
- **`help`:** list of supported commands
- **`model-info`:** get currently selected model card
- **`default-params`:** get default parameters of the Generation Config

You can start conversation by metioning the chatbot `@{chatbot name} {your prompt} {options}`, and the following options are supported.
- **`--top-p {float}`**: determins how many tokens to pick from the top tokens based on the sum of their probabilities(<= `top-p`).
- **`--temperature {float}`**: used to modulate the next token probabilities.
- **`--max-new-tokens {integer}`**: maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
- **`--do-sample`**: determines whether or not to use sampling ; use greedy decoding otherwise.
- **`--max-windows {integer}`**: determines how many past conversations to look up as a reference.
- **`--internet`**: determines whether or not to use internet search capabilities.

If you want to continue conversation based on past conversation histories, you can simply `reply` to chatbot's message. At this time, you don't need to metion its name. However, you need to specify options in every turn. For instance, if you want to `reply` based on internet search information, then you shoul specify `--internet` in your message.
"""
    return help_msg
    
def get_model_info(model_name, model_infos):
    selected_model_info = model_infos[model_name]
    help_msg = f"""## {model_name}
- **Description:** {selected_model_info['desc']}
- **Number of parameters:** {selected_model_info['parameters']}
- **Hugging Face Hub (base):** {selected_model_info['hub(base)']}
- **Hugging Face Hub (ckpt):** {selected_model_info['hub(ckpt)']}
"""
    return help_msg
    
    
def get_default_params(gen_config, max_windows):
    help_msg = f"""{gen_config}, max-windows = {max_windows}"""
    return help_msg