TITLE = "Alpaca-LoRA Playground"

ABSTRACT = """
Thanks to [tolen](https://github.com/tloen/alpaca-lora), this simple application runs Alpaca-LoRA which is instruction fine-tuned version of [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) from Meta AI. Alpaca-LoRA is *Low-Rank LLaMA Instruct-Tuning* which is inspired by [Stanford Alpaca project](https://github.com/tatsu-lab/stanford_alpaca). Alpaca-LoRA is built on the same concept as Standford Alpaca project, but it was trained on a consumer GPU(RTX4090) with [transformers](https://huggingface.co/docs/transformers/index), [peft](https://github.com/huggingface/peft), and [bitsandbytes](https://github.com/TimDettmers/bitsandbytes/tree/main).

I am thankful to the [Jarvislabs.ai](https://jarvislabs.ai/) who generously provided free GPU instances. 
"""

BOTTOM_LINE = """
In order to process batch generation, the common parameters in LLaMA are fixed. If you want to change the values of them, please do that in `generation_config.yaml`
"""

DEFAULT_EXAMPLES = [
    {
        "title": "1️⃣ List all Canadian provinces in alphabetical order.",
        "examples": [
            ["1", "List all Canadian provinces in alphabetical order."],
            ["2", "Which ones are on the east side?"],
            ["3", "What foods are famous in each province?"],
            ["4", "What about sightseeing? or landmarks?"],
        ],
    },
    {
        "title": "2️⃣ Tell me about Alpacas.",
        "examples": [
            ["1", "Tell me about alpacas."],
            ["2", "What other animals are living in the same area?"],
            ["3", "Are they the same species?"],
            ["4", "Write a Python program to return those species"],
        ],
    },
    {
        "title": "3️⃣ Tell me about the king of France in 2019.",
        "examples": [
            ["1", "Tell me about the king of France in 2019."],
        ]
    },
    {
        "title": "4️⃣ Write a Python program that prints the first 10 Fibonacci numbers.",
        "examples": [
            ["1", "Write a Python program that prints the first 10 Fibonacci numbers."],
            ["2", "could you explain how the code works?"]            
        ]
    }
]

SPECIAL_STRS = {
    "continue": "continue.",
    "summarize": "summarize our conversations so far in three sentences."
}