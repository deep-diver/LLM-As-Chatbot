from pingpong.alpaca import AlpacaChatPPManager
from pingpong.koalpaca import KoAlpacaChatPPManager
from pingpong.stablelm import StableLMChatPPManager
from pingpong.flan import FlanAlpacaChatPPManager
from pingpong.os_stablelm import OSStableLMChatPPManager
from pingpong.vicuna import VicunaChatPPManager
from pingpong.stable_vicuna import StableVicunaChatPPManager
from pingpong.starchat import StarChatPPManager
from pingpong.mpt import MPTChatPPManager
from pingpong.baize import BaizeChatPPManager

from pingpong.pingpong import PPManager
from pingpong.pingpong import PromptFmt

class XGenChatPromptFmt(PromptFmt):
    @classmethod
    def ctx(cls, context):
        if context is None or context == "":
            return ""
        else:
            return f"""{context}

"""
    
    @classmethod
    def prompt(cls, pingpong, truncate_size):
        ping = pingpong.ping[:truncate_size]
        pong = "" if pingpong.pong is None else pingpong.pong[:truncate_size]
        return f"""### Human: {ping}
###{pong}
"""
    

class XGenChatPPManager(PPManager):
    def build_prompts(self, from_idx: int=0, to_idx: int=-1, fmt: PromptFmt=XGenChatPromptFmt, truncate_size: int=None):
        if to_idx == -1 or to_idx >= len(self.pingpongs):
            to_idx = len(self.pingpongs)
            
        results = fmt.ctx(self.ctx)
        
        for idx, pingpong in enumerate(self.pingpongs[from_idx:to_idx]):
            results += fmt.prompt(pingpong, truncate_size=truncate_size)
            
        return results    

class OrcaMiniChatPromptFmt(PromptFmt):
    @classmethod
    def ctx(cls, context):
        if context is None or context == "":
            return ""
        else:
            return f"""### System:
{context}
"""
    
    @classmethod
    def prompt(cls, pingpong, truncate_size):
        ping = pingpong.ping[:truncate_size]
        pong = "" if pingpong.pong is None else pingpong.pong[:truncate_size]
        return f"""### User:
{ping}
### Response:
{pong}"""

class OrcaMiniChatPPManager(PPManager):
    def build_prompts(self, from_idx: int=0, to_idx: int=-1, fmt: PromptFmt=OrcaMiniChatPromptFmt, truncate_size: int=None):
        if to_idx == -1 or to_idx >= len(self.pingpongs):
            to_idx = len(self.pingpongs)
            
        results = fmt.ctx(self.ctx)
        
        for idx, pingpong in enumerate(self.pingpongs[from_idx:to_idx]):
            results += fmt.prompt(pingpong, truncate_size=truncate_size)
            
        return results    

class RedPajamaChatPromptFmt(PromptFmt):
    @classmethod
    def ctx(cls, context):
        if context is None or context == "":
            return ""
        else:
            return f"""{context}
"""
    
    @classmethod
    def prompt(cls, pingpong, truncate_size):
        ping = pingpong.ping[:truncate_size]
        pong = "" if pingpong.pong is None else pingpong.pong[:truncate_size]
        return f"""<human>: {ping}
<bot>:{pong}"""

class RedPajamaChatPPManager(PPManager):
    def build_prompts(self, from_idx: int=0, to_idx: int=-1, fmt: PromptFmt=RedPajamaChatPromptFmt, truncate_size: int=None):
        if to_idx == -1 or to_idx >= len(self.pingpongs):
            to_idx = len(self.pingpongs)
            
        results = fmt.ctx(self.ctx)
        
        for idx, pingpong in enumerate(self.pingpongs[from_idx:to_idx]):
            results += fmt.prompt(pingpong, truncate_size=truncate_size)
            
        return results

class GuanacoPromptFmt(PromptFmt):
    @classmethod
    def ctx(cls, context):
        if context is None or context == "":
            return ""
        else:
            return f"""{context}
"""
        
    @classmethod
    def prompt(cls, pingpong, truncate_size):
        ping = pingpong.ping[:truncate_size]
        pong = "" if pingpong.pong is None else pingpong.pong[:truncate_size]
        return f"""### Human: {ping}
### Assistant: {pong}
"""
  
class GuanacoChatPPManager(PPManager):
    def build_prompts(self, from_idx: int=0, to_idx: int=-1, fmt: PromptFmt=GuanacoPromptFmt, truncate_size: int=None):
        if to_idx == -1 or to_idx >= len(self.pingpongs):
            to_idx = len(self.pingpongs)
            
        results = fmt.ctx(self.ctx)
        
        for idx, pingpong in enumerate(self.pingpongs[from_idx:to_idx]):
            results += fmt.prompt(pingpong, truncate_size=truncate_size)
            
        return results 

class WizardPromptFmt(PromptFmt):
    @classmethod
    def ctx(cls, context):
        if context is None or context == "":
            return ""
        else:
            return f"""{context}
"""
        
    @classmethod
    def prompt(cls, pingpong, truncate_size):
        ping = pingpong.ping[:truncate_size]
        pong = "" if pingpong.pong is None else pingpong.pong[:truncate_size]
        return f"""{ping}
### Response: {pong}

"""
  
class WizardChatPPManager(PPManager):
    def build_prompts(self, from_idx: int=0, to_idx: int=-1, fmt: PromptFmt=WizardPromptFmt, truncate_size: int=None):
        if to_idx == -1 or to_idx >= len(self.pingpongs):
            to_idx = len(self.pingpongs)
            
        results = fmt.ctx(self.ctx)
        
        for idx, pingpong in enumerate(self.pingpongs[from_idx:to_idx]):
            results += fmt.prompt(pingpong, truncate_size=truncate_size)
            
        return results

class KULLMPromptFmt(PromptFmt):
    @classmethod
    def ctx(cls, context):
        if context is None or context == "":
            return ""
        else:
            return f"""{context}
"""
        
    @classmethod
    def prompt(cls, pingpong, truncate_size):
        ping = pingpong.ping[:truncate_size]
        pong = "" if pingpong.pong is None else pingpong.pong[:truncate_size]
        return f"""### 명령어:
{ping}
### 응답:
{pong}
"""
  
class KULLMChatPPManager(PPManager):
    def build_prompts(self, from_idx: int=0, to_idx: int=-1, fmt: PromptFmt=KULLMPromptFmt, truncate_size: int=None):
        if to_idx == -1 or to_idx >= len(self.pingpongs):
            to_idx = len(self.pingpongs)
            
        results = fmt.ctx(self.ctx)
        
        for idx, pingpong in enumerate(self.pingpongs[from_idx:to_idx]):
            results += fmt.prompt(pingpong, truncate_size=truncate_size)
            
        return results

def get_chat_manager(model_type):
    if model_type == "alpaca":
        return AlpacaChatPPManager()
    elif model_type == "openllama":
        return AlpacaChatPPManager()
    elif model_type == "alpaca-gpt4":
        return AlpacaChatPPManager()
    elif model_type == "nous-hermes":
        return AlpacaChatPPManager()
    elif model_type == "stablelm":
        return StableLMChatPPManager()
    elif model_type == "os-stablelm":
        return OSStableLMChatPPManager()
    elif model_type == "koalpaca-polyglot":
        return KoAlpacaChatPPManager()
    elif model_type == "kullm-polyglot":
        return KULLMChatPPManager()
    elif model_type == "flan-alpaca":
        return FlanAlpacaChatPPManager()
    elif model_type == "camel":
        return AlpacaChatPPManager()
    elif model_type == "t5-vicuna":
        return VicunaChatPPManager()
    elif model_type == "vicuna":
        return VicunaChatPPManager()
    elif model_type == "stable-vicuna":
        return StableVicunaChatPPManager()
    elif model_type == "starchat":
        return StarChatPPManager()
    elif model_type == "mpt":
        return MPTChatPPManager()
    elif model_type == "redpajama":
        return RedPajamaChatPPManager()
    elif model_type == "llama-deus":
        return AlpacaChatPPManager()
    elif model_type == "evolinstruct-vicuna":
        return VicunaChatPPManager()
    elif model_type == "alpacoom":
        return AlpacaChatPPManager()
    elif model_type == "baize":
        return BaizeChatPPManager()
    elif model_type == "guanaco":
        return GuanacoChatPPManager()
    elif model_type == "falcon":
        return AlpacaChatPPManager()
    elif model_type == "wizard-falcon":
        return WizardChatPPManager()
    elif model_type == "replit-instruct":
        return AlpacaChatPPManager()
    elif model_type == "redpajama-instruct":
        return RedPajamaChatPPManager()
    elif model_type == "airoboros":
        return VicunaChatPPManager()
    elif model_type == "samantha-vicuna":
        return VicunaChatPPManager()
    elif model_type == "lazarus":
        return AlpacaChatPPManager()
    elif model_type == "chronos":
        return AlpacaChatPPManager()
    elif model_type == "wizardlm":
        return VicunaChatPPManager()
    elif model_type == "wizard-vicuna":
        return VicunaChatPPManager()
    elif model_type == "wizard-coder":
        return AlpacaChatPPManager()
    elif model_type == "orcamini":
        return OrcaMiniChatPPManager()
    elif model_type == "xgen":
        return XGenChatPPManager()
    else:
        return None

def get_global_context(model_type):
    if model_type == "xgen":
        return """A chat between a curious human and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the human's questions."""
    elif model_type == "orcamini":
        return """You are an AI assistant that follows instruction extremely well. Help as much as you can.
"""
    elif model_type == "alpaca":
        return """Below are a series of dialogues between human and an AI assistant.
The AI tries to answer the given instruction as in response.
The AI MUST not generate any text containing `### Response` or `### Instruction`.
The AI MUST be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable.
The assistant MUST be happy to help with almost anything, and will do its best to understand exactly what is needed.
It also MUST avoid giving false or misleading information, and it caveats when it isn’t entirely sure about the right answer.
That said, the assistant is practical and really does its best, and doesn’t let caution get too much in the way of being useful.
"""
    elif model_type == "openllama":
        return """Below are a series of dialogues between human and an AI assistant.
The AI tries to answer the given instruction as in response.
The AI MUST not generate any text containing `### Response` or `### Instruction`.
The AI MUST be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable.
The assistant MUST be happy to help with almost anything, and will do its best to understand exactly what is needed.
It also MUST avoid giving false or misleading information, and it caveats when it isn’t entirely sure about the right answer.
That said, the assistant is practical and really does its best, and doesn’t let caution get too much in the way of being useful.
"""
    elif model_type == "alpaca-gpt4":
        return """Below are a series of dialogues between human and an AI assistant.
The AI tries to answer the given instruction as in response.
The AI MUST not generate any text containing `### Response` or `### Instruction`.
The AI MUST be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable.
The assistant MUST be happy to help with almost anything, and will do its best to understand exactly what is needed.
It also MUST avoid giving false or misleading information, and it caveats when it isn’t entirely sure about the right answer.
That said, the assistant is practical and really does its best, and doesn’t let caution get too much in the way of being useful.
"""
    elif model_type == "nous-hermes":
        return """Below are a series of dialogues between human and an AI assistant.
The AI tries to answer the given instruction as in response.
The AI MUST not generate any text containing `### Response` or `### Instruction`.
The AI MUST be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable.
The assistant MUST be happy to help with almost anything, and will do its best to understand exactly what is needed.
It also MUST avoid giving false or misleading information, and it caveats when it isn’t entirely sure about the right answer.
That said, the assistant is practical and really does its best, and doesn’t let caution get too much in the way of being useful.
"""
    elif model_type == "lazarus":
        return """Below are a series of dialogues between human and an AI assistant.
The AI tries to answer the given instruction as in response.
The AI MUST not generate any text containing `### Response` or `### Instruction`.
The AI MUST be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable.
The assistant MUST be happy to help with almost anything, and will do its best to understand exactly what is needed.
It also MUST avoid giving false or misleading information, and it caveats when it isn’t entirely sure about the right answer.
That said, the assistant is practical and really does its best, and doesn’t let caution get too much in the way of being useful.
"""    
    elif model_type == "chronos":
        return """Below are a series of dialogues between human and an AI assistant.
The AI tries to answer the given instruction as in response.
The AI MUST not generate any text containing `### Response` or `### Instruction`.
The AI MUST be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable.
The assistant MUST be happy to help with almost anything, and will do its best to understand exactly what is needed.
It also MUST avoid giving false or misleading information, and it caveats when it isn’t entirely sure about the right answer.
That said, the assistant is practical and really does its best, and doesn’t let caution get too much in the way of being useful.
"""            
    
    elif model_type == "stablelm":
        return """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""
    elif model_type == "os-stablelm":
        return ""
    elif model_type == "koalpaca-polyglot":
        return """아래는 인간과 AI 어시스턴트 간의 일련의 대화입니다.
인공지능은 주어진 질문에 대한 응답으로 대답을 시도합니다.
인공지능은 `### 질문` 또는 `### 응답`가 포함된 텍스트를 생성해서는 안 됩니다.
AI는 도움이 되고, 예의 바르고, 정직하고, 정교하고, 감정을 인식하고, 겸손하지만 지식이 있어야 합니다.
어시스턴트는 거의 모든 것을 기꺼이 도와줄 수 있어야 하며, 무엇이 필요한지 정확히 이해하기 위해 최선을 다해야 합니다.
또한 허위 또는 오해의 소지가 있는 정보를 제공하지 않아야 하며, 정답을 완전히 확신할 수 없을 때는 주의를 환기시켜야 합니다.
즉, 이 어시스턴트는 실용적이고 정말 최선을 다하며 주의를 기울이는 데 너무 많은 시간을 할애하지 않습니다.
"""
    elif model_type == "kullm-polyglot":
        return """아래는 인간과 AI 어시스턴트 간의 일련의 대화입니다.
인공지능은 주어진 명령어에 대한 응답으로 대답을 시도합니다.
인공지능은 `### 명령어` 또는 `### 응답`가 포함된 텍스트를 생성해서는 안 됩니다.
AI는 도움이 되고, 예의 바르고, 정직하고, 정교하고, 감정을 인식하고, 겸손하지만 지식이 있어야 합니다.
어시스턴트는 거의 모든 것을 기꺼이 도와줄 수 있어야 하며, 무엇이 필요한지 정확히 이해하기 위해 최선을 다해야 합니다.
또한 허위 또는 오해의 소지가 있는 정보를 제공하지 않아야 하며, 정답을 완전히 확신할 수 없을 때는 주의를 환기시켜야 합니다.
즉, 이 어시스턴트는 실용적이고 정말 최선을 다하며 주의를 기울이는 데 너무 많은 시간을 할애하지 않습니다.
"""        
    elif model_type == "flan-alpaca":
        return """Below are a series of dialogues between human and an AI assistant.
Each turn of conversation is distinguished by the delimiter of "-----"
The AI MUST be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable.
The assistant MUST be happy to help with almost anything, and will do its best to understand exactly what is needed.
It also MUST avoid giving false or misleading information, and it caveats when it isn’t entirely sure about the right answer.
That said, the assistant is practical and really does its best, and doesn’t let caution get too much in the way of being useful.
"""
    elif model_type == "camel":
        return """Below are a series of dialogues between human and an AI assistant.
The AI tries to answer the given instruction as in response.
The AI MUST not generate any text containing `### Response` or `### Instruction`.
The AI MUST be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable.
The assistant MUST be happy to help with almost anything, and will do its best to understand exactly what is needed.
It also MUST avoid giving false or misleading information, and it caveats when it isn’t entirely sure about the right answer.
That said, the assistant is practical and really does its best, and doesn’t let caution get too much in the way of being useful.
"""
    elif model_type == "t5-vicuna":
        return """A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions.
"""
    elif model_type == "vicuna":
        return """A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions.
"""
    elif model_type == "airoboros":
        return """A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions.
"""        
    elif model_type == "stable-vicuna":
        return """A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions.
"""
    elif model_type == "wizardlm":
        return """A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions.
"""
    elif model_type == "wizard-vicuna":
        return """A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions.
"""
    elif model_type == "starchat":
        return """Below is a conversation between a human user and a helpful AI coding assistant.
"""
    elif model_type == "mpt":
        return """<|im_start|>system
- You are a helpful assistant chatbot trained by MosaicML.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.<|im_end|>
"""
    elif model_type == "redpajama":
        return ""
    elif model_type == "redpajama-instruct":
        return ""
    elif model_type == "llama-deus":
        return """Below are a series of dialogues between human and an AI assistant.
The AI tries to answer the given instruction as in response.
The AI MUST not generate any text containing `### Response` or `### Instruction`.
The AI MUST be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable.
The assistant MUST be happy to help with almost anything, and will do its best to understand exactly what is needed.
It also MUST avoid giving false or misleading information, and it caveats when it isn’t entirely sure about the right answer.
That said, the assistant is practical and really does its best, and doesn’t let caution get too much in the way of being useful.
"""
    elif model_type == "evolinstruct-vicuna":
        return """A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions.
"""
    elif model_type == "alpacoom":
        return """Below are a series of dialogues between human and an AI assistant.
The AI tries to answer the given instruction as in response.
The AI MUST not generate any text containing `### Response` or `### Instruction`.
The AI MUST be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable.
The assistant MUST be happy to help with almost anything, and will do its best to understand exactly what is needed.
It also MUST avoid giving false or misleading information, and it caveats when it isn’t entirely sure about the right answer.
That said, the assistant is practical and really does its best, and doesn’t let caution get too much in the way of being useful.
"""
    elif model_type == "baize":
        return """The following is a conversation between a human and an AI assistant named Baize (named after a mythical creature in Chinese folklore). Baize is an open-source AI assistant developed by UCSD and Sun Yat-Sen University. The human and the AI assistant take turns chatting. Human statements start with [|Human|] and AI assistant statements start with [|AI|]. The AI assistant always provides responses in as much detail as possible, and in Markdown format. The AI assistant always declines to engage with topics, questions and instructions related to unethical, controversial, or sensitive issues. Complete the transcript in exactly that format.\n[|Human|]Hello!\n[|AI|]Hi!
"""
    elif model_type == "guanaco":
        return """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
"""
    elif model_type == "falcon":
        return """Below are a series of dialogues between human and an AI assistant.
The AI tries to answer the given instruction as in response.
The AI MUST not generate any text containing `### Response` or `### Instruction`.
The AI MUST be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable.
The assistant MUST be happy to help with almost anything, and will do its best to understand exactly what is needed.
It also MUST avoid giving false or misleading information, and it caveats when it isn’t entirely sure about the right answer.
That said, the assistant is practical and really does its best, and doesn’t let caution get too much in the way of being useful.
"""
    elif model_type == "wizard_falcon":
        return ""
    elif model_type == "samantha-vicuna":
        return "You are Samantha, a sentient AI."
    elif model_type == "wizard-coder":
        return "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    else:
        return ""    