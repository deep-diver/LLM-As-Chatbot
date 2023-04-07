class ConstraintsConfig:
    def __init__(self, constraints):
        for k, v in constraints.items():
            setattr(self, k, v)
            
    def len_exceed(self, context, prompt):
        if len(context) > self.max_context or \
            len(prompt) > self.max_prompt:
            return True
        
        return False
    
    def conv_len_exceed(self, conv_length):
        if conv_length > self.max_conv_len:
            return True
        
        return False