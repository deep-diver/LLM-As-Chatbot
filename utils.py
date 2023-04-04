from chats import alpaca
from chats import baize

def get_chat_interface(model_type, batch_enabled):
    match model_type:
        case 'alpaca':
            return alpaca.chat_batch if batch_enabled else alpaca.chat_stream
        case 'baize':
            return baize.chat_batch if batch_enabled else baize.chat_stream
        case other:
            return None
