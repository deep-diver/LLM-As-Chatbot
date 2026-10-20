"""Exercise the real dispatcher/formatter with only model-loading boundaries stubbed.

Install bingbong without its GPU/embedding dependencies for these tests:
    pip install --no-deps bingbong==0.4.3
    pip install requests sseclient-py
"""
import importlib
import inspect
import json
import sys
import types
from contextlib import contextmanager
from unittest.mock import patch

from pingpong.gradio import GradioAlpacaChatPPManager


@contextmanager
def runtime():
    globals_module = types.ModuleType("global_vars")
    globals_module.remote_addr = None
    globals_module.remote_port = None
    globals_module.remote_token = None
    globals_module.model_name = "scripted-model"
    globals_module.model_thumbnail_tiny = "https://example.com/model.png"
    prompts = []
    pre = types.ModuleType("chats.pre")

    def build(prompt, *args, **kwargs):
        prompts.append(prompt)
        return {}, iter(["Scripted model response."])

    pre.build = build
    pre.start_gen = lambda kwargs: None
    # This is the unchanged helper used by build_prompts, independent of a model.
    import re
    pre.contains_image_markdown = lambda text: re.search(r'!\[(.*?)\]\((.*?)\)', text)
    batch = types.ModuleType("gens.batch_gen")
    batch.get_output_batch = lambda *args, **kwargs: None
    modules = {"global_vars": globals_module, "chats.pre": pre, "gens.batch_gen": batch}
    # central imports all model adapters; only Alpaca is needed for this caller test.
    for name in ("stablelm", "koalpaca", "flan_alpaca", "os_stablelm", "vicuna",
                 "stable_vicuna", "starchat", "wizard_coder", "redpajama", "mpt",
                 "alpacoom", "baize", "guanaco", "falcon", "wizard_falcon", "xgen",
                 "llama2", "freewilly", "mistral", "custom"):
        modules["chats." + name] = types.ModuleType("chats." + name)
    with patch.dict(sys.modules, modules):
        central = importlib.import_module("chats.central")
        yield types.SimpleNamespace(central=central, globals=globals_module, prompts=prompts)


def state_and_history():
    state = {"ppmanager_type": GradioAlpacaChatPPManager, "model_type": "alpaca"}
    history = [json.loads(str(GradioAlpacaChatPPManager())) for _ in range(2)]
    return state, history


def chat_arguments(central, state, history, question="What is Python?", mode="on", key="", index=0):
    arguments = {name: None for name in inspect.signature(central.sync_chat_stream).parameters}
    arguments.update(idx=index, local_data=history, user_message=question, state=state,
                     global_context="Be helpful.", ctx_num_lconv=3, internet_option=mode,
                     serper_api_key=key)
    return arguments
