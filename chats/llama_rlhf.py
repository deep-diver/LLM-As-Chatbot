from threading import Thread
import global_vars

from transformers import TextIteratorStreamer
import gradio as gr

from gens.batch_gen import get_output_batch
from miscs.strings import SPECIAL_STRS
from miscs.constants import num_of_characters_to_keep
from miscs.utils import common_post_process, post_processes_batch, post_process_stream

from chats.prompts import generate_prompt

def chat_stream(
    context,
    instruction,
    state_chatbot,
):
    if global_vars.constraints_config.len_exceed(context, instruction):
        raise gr.Error("context or prompt is too long!")
    
    bot_summarized_response = ''
    # user input should be appropriately formatted (don't be confused by the function name)
    instruction_display = common_post_process(instruction)
    instruction_prompt, conv_length = generate_prompt(instruction, state_chatbot, context, "Question:", "Answer:")
    
    if global_vars.constraints_config.conv_len_exceed(conv_length):
        instruction_prompt = generate_prompt(SPECIAL_STRS["summarize"], state_chatbot, context, "Question:", "Answer:", partial=True)[0]
        
        state_chatbot = state_chatbot + [
            (
                None, 
                "![](https://s2.gifyu.com/images/icons8-loading-circle.gif) too long conversations, so let's summarize..."
            )
        ]
        yield (state_chatbot, state_chatbot, context)
        
        bot_summarized_response = get_output_batch(
            global_vars.model, global_vars.tokenizer, [instruction_prompt], global_vars.gen_config_summarization
        )[0]
        bot_summarized_response = bot_summarized_response.split("### Response:")[-1].strip()
        
        state_chatbot[-1] = (
            None, 
            "✅ summarization is done and set as context"
        )
        print(f"bot_summarized_response: {bot_summarized_response}")
        yield (state_chatbot, state_chatbot, f"{context}. {bot_summarized_response}".strip())
        
    instruction_prompt = generate_prompt(instruction, state_chatbot, f"{context} {bot_summarized_response}", "Question:", "Answer:")[0]
    
    model_inputs = global_vars.tokenizer([instruction_prompt], return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(global_vars.tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)    
    
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
    )
    generate_kwargs.update(global_vars.gen_config_raw)
    
    instruction_display = None if instruction_display == SPECIAL_STRS["continue"] else instruction_display
    state_chatbot = state_chatbot + [(instruction_display, None)]
    yield (state_chatbot, state_chatbot, f"{context}. {bot_summarized_response}".strip())

    t = Thread(target=global_vars.model.generate, kwargs=generate_kwargs)
    t.start()    
    
    agg_tokens = ""
    for new_text in streamer:
        agg_tokens += new_text
        state_chatbot[-1] = (instruction_display, agg_tokens)
        yield (state_chatbot, state_chatbot, f"{context} {bot_summarized_response}".strip())

    yield (
        state_chatbot,
        state_chatbot,
        f"{context} {bot_summarized_response}".strip()
    )


def chat_batch(
    contexts,
    instructions, 
    state_chatbots,
):
    state_results = []
    ctx_results = []

    instruct_prompts = [
        generate_prompt(instruct, histories, ctx)[0]
        for ctx, instruct, histories in zip(contexts, instructions, state_chatbots)
    ]
        
    bot_responses = get_output_batch(
        global_vars.model, global_vars.tokenizer, instruct_prompts, global_vars.gen_config
    )
    bot_responses = post_processes_batch(bot_responses)

    for ctx, instruction, bot_response, state_chatbot in zip(contexts, instructions, bot_responses, state_chatbots):
        new_state_chatbot = state_chatbot + [('' if instruction == SPECIAL_STRS["continue"] else instruction, bot_response)]
        ctx_results.append(gr.Textbox.update(value=bot_response) if instruction == SPECIAL_STRS["summarize"] else ctx)
        state_results.append(new_state_chatbot)

    return (state_results, state_results, ctx_results)