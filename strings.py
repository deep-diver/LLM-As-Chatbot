TITLE = "Alpaca-LoRA Playground"

ABSTRACT = """
Thanks to [tolen](https://github.com/tloen/alpaca-lora), this simple application runs Alpaca-LoRA which is instruction fine-tuned version of [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) from Meta AI. Alpaca-LoRA is *Low-Rank LLaMA Instruct-Tuning* which is inspired by [Stanford Alpaca project](https://github.com/tatsu-lab/stanford_alpaca). Alpaca-LoRA is built on the same concept as Standford Alpaca project, but it was trained on a consumer GPU(RTX4090) with [transformers](https://huggingface.co/docs/transformers/index), [peft](https://github.com/huggingface/peft), and [bitsandbytes](https://github.com/TimDettmers/bitsandbytes/tree/main).

I am thankful to the [Jarvislabs.ai](https://jarvislabs.ai/) who generously provided free GPU instances. 
"""

BOTTOM_LINE = """
In order to process batch generation, the common parameters in LLaMA are fixed as below:
- `temperature=0.90`
- `top_p=0.75`
- `num_beams=2`
"""

DEFAULT_EXAMPLES = [
    ["1️⃣", "List all Canadian provinces in alphabetical order."],
    ["1️⃣ ▶️ 1️⃣", "Which ones are on the east side?"],
    ["1️⃣ ▶️ 2️⃣", "What foods are famous in each province?"],
    ["1️⃣ ▶️ 3️⃣", "What about sightseeing? or landmarks?"],
    ["2️⃣", "Tell me about alpacas."],
    ["2️⃣ ▶️ 1️⃣", "What other animals are living in the same area?"],
    ["2️⃣ ▶️ 2️⃣", "Are they the same species?"],
    ["2️⃣ ▶️ 3️⃣", "Write a Python program to return those species"],
    ["3️⃣", "Tell me about the king of France in 2019."],                
    ["4️⃣", "Write a Python program that prints the first 10 Fibonacci numbers."],                
]