class ConvContext:
    def __init__(self, ctx):
        self._ctx = ctx

    def set_ctx(self, context, append=False):
        pass
        
    @property
    def prompt_fmt(self):
        pass
    
class AlpacaConvContext(ConvContext):
    def set_ctx(self, context, append=False):
        self._ctx = f"{self._ctx} {context}" if append else context    
    
    @property
    def prompt_fmt(self):
        return f"""{self._ctx}

"""    

class PingPong:
    def __init__(self, ping, pong):
        self._ping = ping
        self._pong = pong
        
    def set_ping(self, ping):
        self._ping = ping
    
    def set_pong(self, pong):
        self._pong = pong
        
    @property
    def prompt_fmt(self):
        pass
    
    @property
    def ui_fmt(self):
        pass
    
    
class AlpacaPingPong(PingPong):
    @property
    def prompt_fmt(self):
        if self._ping is None:
            return f"""{self._pong}
            
"""
        else:
            return f"""### Instruction:{self._ping}

    ### Response:{self._pong}

    """
    
    @property
    def ui_fmt(self):
        return (self._ping, self._pong)
    
################################################
    
class ConvManager:
    def __init__(self, ctx, context_max, prompt_max):
        self._ctx = ctx
        self._contex_max = context_max
        self._prompt_max = prompt_max
        self._histories = []
        
    def add_conversation(self, pingpong):
        self._histories.append(pingpong)

    def set_ctx(self, context, append=False):
        self._ctx.set_ctx(context, append)
        
    def get_prompt_fmt(self, last_k=1):
        conv = self._ctx.prompt_fmt
        sub_convs = ""
        
        if len(self._histories) > 0:
            for history in self._histories[-last_k:]:
                sub_convs = sub_convs + history.prompt_fmt
            
        conv = conv + sub_convs
        return conv, sub_convs
    
    # currently gradio natively
    def get_ui_fmt(self, last_k=None):
        if last_k is None:
            histories = self._histories
        else:
            histories = self._histories[-last_k:]
        
        conv = []
        
        if len(self._histories) > 0:
            for history in histories:
                conv.append(history.ui_fmt)
            
        return conv