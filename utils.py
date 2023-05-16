from chats import alpaca
from chats import alpaca_gpt4
from chats import stablelm
from chats import koalpaca
from chats import os_stablelm
from chats import vicuna
from chats import flan_alpaca
from chats import starchat
from chats import redpajama
from chats import mpt

from pingpong.gradio import GradioAlpacaChatPPManager
from pingpong.gradio import GradioKoAlpacaChatPPManager
from pingpong.gradio import GradioStableLMChatPPManager
from pingpong.gradio import GradioFlanAlpacaChatPPManager
from pingpong.gradio import GradioOSStableLMChatPPManager
from pingpong.gradio import GradioVicunaChatPPManager
from pingpong.gradio import GradioStableVicunaChatPPManager
from pingpong.gradio import GradioStarChatPPManager
from pingpong.gradio import GradioMPTChatPPManager
from pingpong.gradio import GradioRedPajamaChatPPManager

def get_chat_interface(model_type):
    if model_type == "alpaca":
        return alpaca.chat_stream
    elif model_type == "alpaca-gpt4":
        return alpaca.chat_stream
    elif model_type == "stablelm":
        return stablelm.chat_stream
    elif model_type == "os-stablelm":
        return os_stablelm.chat_stream
    elif model_type == "koalpaca-polyglot":
        return koalpaca.chat_stream
    elif model_type == "flan-alpaca":
        return flan_alpaca.chat_stream
    elif model_type == "camel":
        return alpaca.chat_stream
    elif model_type == "t5-vicuna":
        return vicuna.chat_stream
    elif model_type == "stable-vicuna":
        return vicuna.chat_stream
    elif model_type == "starchat":
        return starchat.chat_stream
    elif model_type == "mpt":
        return mpt.chat_stream
    elif model_type == "redpajama":
        return redpajama.chat_stream
    elif model_type == "vicuna":
        return vicuna.chat_stream    
    else:
        return None

def get_chat_manager(model_type):
    if model_type == "alpaca":
        return GradioAlpacaChatPPManager()
    elif model_type == "alpaca-gpt4":
        return GradioAlpacaChatPPManager()
    elif model_type == "stablelm":
        return GradioStableLMChatPPManager()
    elif model_type == "os-stablelm":
        return GradioOSStableLMChatPPManager()
    elif model_type == "koalpaca-polyglot":
        return GradioKoAlpacaChatPPManager()
    elif model_type == "flan-alpaca":
        return GradioFlanAlpacaChatPPManager()
    elif model_type == "camel":
        return GradioAlpacaChatPPManager()
    elif model_type == "t5-vicuna":
        return GradioVicunaChatPPManager()
    elif model_type == "vicuna":
        return GradioVicunaChatPPManager()
    elif model_type == "stable-vicuna":
        return GradioStableVicunaChatPPManager()
    elif model_type == "starchat":
        return GradioStarChatPPManager()
    elif model_type == "mpt":
        return GradioMPTChatPPManager()
    elif model_type == "redpajama":
        return GradioRedPajamaChatPPManager()
    else:
        return None