import os
import argparse

from discord_app import discord_main
from app import gradio_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    app_mode = os.getenv("LLMCHAT_APP_MODE")
    if app_mode is None or \
        app_mode not in ["GRADIO", "DISCORD"]:
        app_mode = "GRADIO"
    
    if app_mode == "GRADIO":
        parser.add_argument('--root-path', default="")
        parser.add_argument('--local-files-only', default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument('--share', default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument('--debug', default=False, action=argparse.BooleanOptionalAction)
        args = parser.parse_args()
        gradio_main(args)
        
    elif app_mode == "DISCORD":
        parser.add_argument('--token', default=None, type=str)
        parser.add_argument('--model-name', default=None, type=str) 
        parser.add_argument('--max-workers', default=1, type=int)
        parser.add_argument('--mode-cpu', default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument('--mode-mps', default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument('--mode-8bit', default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument('--mode-4bit', default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument('--mode-full-gpu', default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--local-files-only', default=False, action=argparse.BooleanOptionalAction)
        args = parser.parse_args()
        discord_main(args)