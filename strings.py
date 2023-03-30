TITLE = "Alpaca-LoRA Playground"

ABSTRACT = """
Thanks to [tolen](https://github.com/tloen/alpaca-lora), this application runs Alpaca-LoRA which is instruction fine-tuned version of [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/). This demo currently runs 30B version on a 3*A6000 instance at [Jarvislabs.ai](https://jarvislabs.ai/).

NOTE: too long input (context, instruction) will not be allowed. Please keep them < 150
"""

BOTTOM_LINE = """
This demo application runs the open source project, [Alpaca-LoRA-Serve](https://github.com/deep-diver/Alpaca-LoRA-Serve). By default, it runs with streaming mode, but you can also run with dynamic batch generation model. Please visit the repo, find more information, and contribute if you can.

Alpaca-LoRA is built on the same concept as Standford Alpaca project, but it lets us train and inference on a smaller GPUs such as RTX4090 for 7B version. Also, we could build very small size of checkpoints on top of base models thanks to [🤗 transformers](https://huggingface.co/docs/transformers/index), [🤗 peft](https://github.com/huggingface/peft), and [bitsandbytes](https://github.com/TimDettmers/bitsandbytes/tree/main) libraries.

We are thankful to the [Jarvislabs.ai](https://jarvislabs.ai/) who generously provided free GPU instances.
"""

DEFAULT_EXAMPLES = {
    "Typical Questions": [
        {
            "title": "List all Canadian provinces in alphabetical order.",
            "examples": [
                ["1", "List all Canadian provinces in alphabetical order."],
                ["2", "Which ones are on the east side?"],
                ["3", "What foods are famous in each province on the east side?"],
                ["4", "What about sightseeing? or landmarks? list one per province"],
            ],
        },
        {
            "title": "Tell me about Alpacas.",
            "examples": [
                ["1", "Tell me about alpacas in two sentences"],
                ["2", "What other animals are living in the same area?"],
                ["3", "Are they the same species?"],
                ["4", "Write a Python program to return those species"],
            ],
        },
        {
            "title": "Tell me about the king of France in 2019.",
            "examples": [
                ["1", "Tell me about the king of France in 2019."],
                ["2", "What about before him?"],
            ]
        },
        {
            "title": "Write a Python program that prints the first 10 Fibonacci numbers.",
            "examples": [
                ["1", "Write a Python program that prints the first 10 Fibonacci numbers."],
                ["2", "Could you explain how the code works?"],
                ["3", "What is recursion?"],
            ]
        }
    ],
    "Identity": [
        {
            "title": "Conversation with the planet Pluto",
            "examples": [
                ["1", "Conversation with the planet Pluto", "I'am so curious about you"],
                ["2", "Conversation with the planet Pluto", "Tell me what I would see if I visited"],
                ["3", "Conversation with the planet Pluto", "It sounds beautiful"],
                ["4", "Conversation with the planet Pluto", "I'll keep that in mind. Hey I was wondering have you ever had any visitor?"],
                ["5", "Conversation with the planet Pluto", "That must have been exciting"],
                ["6", "Conversation with the planet Pluto", "That's so great. What else do you wish people knew about you?"],
                ["7", "Conversation with the planet Pluto", "Thanks for talking with me"],
            ]
        },
        {
            "title": "Conversation with a paper airplane",
            "examples": [
                ["1", "Conversation with a paper airplane", "What's it like being thrown through the air"],
                ["2", "Conversation with a paper airplane", "What's the worst place you've ever landed"],
                ["3", "Conversation with a paper airplane", "Have you ever stucked?"],
                ["4", "Conversation with a paper airplane", "What's the secret to a really good paper airplane?"],
                ["5", "Conversation with a paper airplane", "What's the farthest you've ever flown?"],
                ["6", "Conversation with a paper airplane", "Good to talk to you!"]
            ]
        }
    ]
}

SPECIAL_STRS = {
    "continue": "continue.",
    "summarize": "summarize our conversations so far in three sentences."
}