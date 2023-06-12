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
from chats import alpacoom
from chats import baize
from chats import guanaco
from chats import falcon

from pingpong.gradio import GradioAlpacaChatPPManager
from pingpong.gradio import GradioKoAlpacaChatPPManager
from pingpong.gradio import GradioStableLMChatPPManager
from pingpong.gradio import GradioFlanAlpacaChatPPManager
from pingpong.gradio import GradioOSStableLMChatPPManager
from pingpong.gradio import GradioVicunaChatPPManager
from pingpong.gradio import GradioStableVicunaChatPPManager
from pingpong.gradio import GradioStarChatPPManager
from pingpong.gradio import GradioMPTChatPPManager
from pingpong.gradio import GradioBaizeChatPPManager

from pingpong.pingpong import PPManager
from pingpong.pingpong import PromptFmt
from pingpong.pingpong import UIFmt
from pingpong.gradio import GradioChatUIFmt

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

class GradioRedPajamaChatPPManager(RedPajamaChatPPManager):
    def build_uis(self, from_idx: int=0, to_idx: int=-1, fmt: UIFmt=GradioChatUIFmt):
        if to_idx == -1 or to_idx >= len(self.pingpongs):
            to_idx = len(self.pingpongs)
        
        results = []
        
        for pingpong in self.pingpongs[from_idx:to_idx]:
            results.append(fmt.ui(pingpong))
            
        return results
    
class RedPajamaInstructPromptFmt(PromptFmt):
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
        return f"""Q: {ping}
A:{pong}"""
  
class RedPajamaInstructChatPPManager(PPManager):
    def build_prompts(self, from_idx: int=0, to_idx: int=-1, fmt: PromptFmt=RedPajamaInstructPromptFmt, truncate_size: int=None):
        if to_idx == -1 or to_idx >= len(self.pingpongs):
            to_idx = len(self.pingpongs)
            
        results = fmt.ctx(self.ctx)
        
        for idx, pingpong in enumerate(self.pingpongs[from_idx:to_idx]):
            results += fmt.prompt(pingpong, truncate_size=truncate_size)
            
        return results

class GradioRedPajamaInstructChatPPManager(RedPajamaInstructChatPPManager):
    def build_uis(self, from_idx: int=0, to_idx: int=-1, fmt: UIFmt=GradioChatUIFmt):
        if to_idx == -1 or to_idx >= len(self.pingpongs):
            to_idx = len(self.pingpongs)
        
        results = []
        
        for pingpong in self.pingpongs[from_idx:to_idx]:
            results.append(fmt.ui(pingpong))
            
        return results

###

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

class GradioGuanacoChatPPManager(GuanacoChatPPManager):
    def build_uis(self, from_idx: int=0, to_idx: int=-1, fmt: UIFmt=GradioChatUIFmt):
        if to_idx == -1 or to_idx >= len(self.pingpongs):
            to_idx = len(self.pingpongs)
        
        results = []
        
        for pingpong in self.pingpongs[from_idx:to_idx]:
            results.append(fmt.ui(pingpong))
            
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

class GradioWizardChatPPManager(WizardChatPPManager):
    def build_uis(self, from_idx: int=0, to_idx: int=-1, fmt: UIFmt=GradioChatUIFmt):
        if to_idx == -1 or to_idx >= len(self.pingpongs):
            to_idx = len(self.pingpongs)
        
        results = []
        
        for pingpong in self.pingpongs[from_idx:to_idx]:
            results.append(fmt.ui(pingpong))
            
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

class GradioKULLMChatPPManager(KULLMChatPPManager):
    def build_uis(self, from_idx: int=0, to_idx: int=-1, fmt: UIFmt=GradioChatUIFmt):
        if to_idx == -1 or to_idx >= len(self.pingpongs):
            to_idx = len(self.pingpongs)
        
        results = []
        
        for pingpong in self.pingpongs[from_idx:to_idx]:
            results.append(fmt.ui(pingpong))
            
        return results

def get_chat_manager(model_type):
    if model_type == "alpaca":
        return GradioAlpacaChatPPManager()
    elif model_type == "alpaca-gpt4":
        return GradioAlpacaChatPPManager()
    elif model_type == "nous-hermes":
        return GradioAlpacaChatPPManager()
    elif model_type == "stablelm":
        return GradioStableLMChatPPManager()
    elif model_type == "os-stablelm":
        return GradioOSStableLMChatPPManager()
    elif model_type == "koalpaca-polyglot":
        return GradioKoAlpacaChatPPManager()
    elif model_type == "kullm-polyglot":
        return GradioKULLMChatPPManager()
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
    elif model_type == "llama-deus":
        return GradioAlpacaChatPPManager()
    elif model_type == "evolinstruct-vicuna":
        return GradioVicunaChatPPManager()
    elif model_type == "alpacoom":
        return GradioAlpacaChatPPManager()
    elif model_type == "baize":
        return GradioBaizeChatPPManager()
    elif model_type == "guanaco":
        return GradioGuanacoChatPPManager()
    elif model_type == "falcon":
        return GradioAlpacaChatPPManager()
    elif model_type == "wizard-falcon":
        return GradioWizardChatPPManager()
    elif model_type == "replit-instruct":
        return GradioAlpacaChatPPManager()
    elif model_type == "redpajama-instruct":
        return GradioRedPajamaChatPPManager()
    elif model_type == "airoboros":
        return GradioVicunaChatPPManager()
    elif model_type == "samantha-vicuna":
        return GradioVicunaChatPPManager()
    elif model_type == "lazarus":
        return GradioAlpacaChatPPManager()
    elif model_type == "chronos":
        return GradioAlpacaChatPPManager()
    elif model_type == "wizardlm":
        return GradioVicunaChatPPManager()
    else:
        return None

def get_global_context(model_type):
    if model_type == "alpaca":
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
    else:
        return ""    