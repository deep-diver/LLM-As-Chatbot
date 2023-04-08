from functools import partial
import global_vars

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
    
    gen_prompt = partial(generate_prompt, ctx_indicator="### Input:", user_indicator="### Instruction:", ai_indicator="### Response:")
    bot_summarized_response = ''
    
    # user input should be appropriately formatted (don't be confused by the function name)
    instruction_display = common_post_process(instruction)
    instruction_prompt, conv_length = gen_prompt(instruction, state_chatbot, context)
    
    if global_vars.constraints_config.conv_len_exceed(conv_length):
        instruction_prompt = gen_prompt(SPECIAL_STRS["summarize"], state_chatbot, context, partial=True)[0]
        
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
        
    instruction_prompt = gen_prompt(instruction, state_chatbot, f"{context} {bot_summarized_response}")[0]
    print(instruction_prompt)
    
    bot_response = global_vars.stream_model(
        instruction_prompt,
        max_tokens=256,
        temperature=1,
        top_p=0.9
    )
    
    instruction_display = None if instruction_display == SPECIAL_STRS["continue"] else instruction_display
    state_chatbot = state_chatbot + [(instruction_display, None)]
    yield (state_chatbot, state_chatbot, f"{context}. {bot_summarized_response}".strip())
    
    prev_index = 0
    agg_tokens = ""
    cutoff_idx = 0
    for tokens in bot_response:
        tokens = tokens.strip()
        cur_token = tokens[prev_index:]
        
        if "#" in cur_token and agg_tokens == "":
            cutoff_idx = tokens.find("#")
            agg_tokens = tokens[cutoff_idx:]

        if agg_tokens != "":
            if len(agg_tokens) < len("### Instruction:") :
                agg_tokens = agg_tokens + cur_token
            elif len(agg_tokens) >= len("### Instruction:"):
                if tokens.find("### Instruction:") > -1:
                    processed_response, _ = post_process_stream(tokens[:tokens.find("### Instruction:")].strip())

                    state_chatbot[-1] = (
                        instruction_display, 
                        processed_response
                    )
                    yield (state_chatbot, state_chatbot, f"{context} {bot_summarized_response}".strip())
                    break
                else:
                    agg_tokens = ""
                    cutoff_idx = 0

        if agg_tokens == "":
            processed_response, to_exit = post_process_stream(tokens)
            state_chatbot[-1] = (instruction_display, processed_response)
            yield (state_chatbot, state_chatbot, f"{context} {bot_summarized_response}".strip())

            if to_exit:
                break

        prev_index = len(tokens)

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