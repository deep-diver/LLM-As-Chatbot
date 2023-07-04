import argparse
import sys

def parse_req(message, gen_config_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-new-tokens', default=256, type=int)
    parser.add_argument('--temperature', default=0.8, type=float)
    parser.add_argument('--max-windows', default=3, type=int)
    parser.add_argument('--do-sample', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--top-p', default=0.75, type=float)
    
    msg = message.strip()
    multiparts_msg = msg.split("--")

    err_msg = None
    user_msg = multiparts_msg[0].strip()
    options = '--'.join(multiparts_msg[1:]).split(' ')
    print(f"options = {options}")
    if options[0] == '': options = []
    else: options[0] = f"--{options[0]}"
    
    try:
        args = parser.parse_args(options)
    except SystemExit as e:
        try:
            args, ex_args = parser.parse_known_args(options)
            err_msg = f"> **ERROR:** some of unrecognized flags({ex_args}) found. allowed flags(--max-new-token, --temperature, --max-windows, --do-sample, --top-p). fall back to default"
        except SystemExit as e:
            args = parser.parse_args([])
            err_msg = f"> **ERROR:** allowed flags(--max-new-token, --temperature, --max-windows, --do-sample, --top-p). fall back to default."
        
    return user_msg, args, err_msg
