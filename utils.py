from chats import alpaca
from chats import baize

def get_chat_interface(model_type, batch_enabled):
    if model_type == "alpaca":
        return alpaca.chat_batch if batch_enabled else alpaca.chat_stream
    elif model_type == "baize":
        return baize.chat_batch if batch_enabled else baize.chat_stream
    else:
        return None
