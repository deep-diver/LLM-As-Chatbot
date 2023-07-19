import argparse
import sys

known_flags_def = {
    "max-new-tokens": {
        "default": None,
        "type": int
    },
    "temperature": {
        "default": None,
        "type": float
    },
    "max-windows": {
        "default": 3,
        "type": int
    },
    "do-sample": {
        "default": True,
        "type": bool
    },
    "top-p": {
        "default": None,
        "type": float
    },
    "internet": {
        "default": False,
        "type": bool
    }
}

def parse_req(message, gen_config):
    message, flags = parse_known_flags(
        message, 
        known_flags_def,
        gen_config
    )
    return message, flags

def init_flags(known_flags_def, gen_config):
    gen_config_attrs = vars(gen_config)
    known_flags = list(known_flags_def.keys())
    flags = {}
    types = {}

    for known_flag in known_flags:
        flags[known_flag] = known_flags_def[known_flag]['default']
        types[known_flag] = known_flags_def[known_flag]['type']
        
        known_flag_underscore = known_flag.replace("-", "_")
        if known_flag_underscore in list(gen_config_attrs.keys()):
            if gen_config_attrs[known_flag_underscore] is not None:
                flags[known_flag] = gen_config_attrs[known_flag_underscore]

    return known_flags, flags, types

def parse_known_flags(string, known_flags_def, gen_config, prefix="--"):
    words = string.split()
    known_flags, flags, types = init_flags(known_flags_def, gen_config)

    for i in range(len(words)):
        word = words[i]
        if word.startswith(prefix):
            flag = word[2:]
            if flag in known_flags:
                if types[flag] == bool:
                    flags[flag] = True
                else:
                    flags[flag] = None

                value = words[i+1:i+2]
                if len(value) != 0:
                    value = value[0]
                    try:
                        flags[flag] = types[flag](value)
                    except ValueError:
                        continue
                    i = i+1

    for k, v in flags.items():
        sub_str = f"{prefix}{k}"
        if v is not None:
            if not isinstance(v, bool):    
                sub_str = sub_str + " " + str(v)
        
        print(sub_str)
        string = string.replace(sub_str, "")

    return string.strip(), flags