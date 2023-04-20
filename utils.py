from chats import alpaca
from chats import alpaca_gpt4
from chats import stablelm
from pingpong.gradio import GradioAlpacaChatPPManager
from pingpong.gradio import GradioStableLMChatPPManager

def get_chat_interface(model_type):
    if model_type == "alpaca":
        return alpaca.chat_stream
    elif model_type == "alpaca-gpt4":
        return alpaca_gpt4.chat_stream
    elif model_type == "stablelm":
        return stablelm.chat_stream
    else:
        return None

def get_chat_manager(model_type):
    if model_type == "alpaca":
        return GradioAlpacaChatPPManager()
    elif model_type == "alpaca-gpt4":
        return GradioAlpacaChatPPManager()
    elif model_type == "stablelm":
        return GradioStableLMChatPPManager()
    else:
        return None    