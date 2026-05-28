import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


LLAMA_PROMPT = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)
QWEN_PROMPT = (
    "<|im_start|>user\n{prompt}\n<|im_end|>\n"
    "<|im_start|>assistant\n"
)

LLAMA_NO_SYSTEM_PROMPT_CHAT_TEMPLATE = """{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- set date_string = "26 Jul 2024" %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
    {%- set has_system_message = true %}
{%- else %}
    {%- set system_message = "" %}
    {%- set has_system_message = false %}
{%- endif %}
{%- if has_system_message %}
{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}
{%- if builtin_tools is defined or tools is not none %}
    {{- "Environment: ipython\\n" }}
{%- endif %}
{%- if builtin_tools is defined %}
    {{- "Tools: " + builtin_tools | reject('equalto', 'code_interpreter') | join(", ") + "\\n\\n"}}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\\n" }}
{{- "Today Date: " + date_string + "\\n\\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\\n\\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\\n\\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}
{%- endif %}
{%- if tools_in_user_message and not tools is none %}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\\n\\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\\n\\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}
{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}
            {{- "<|python_tag|>" + tool_call.name + ".call(" }}
            {%- for arg_name, arg_val in tool_call.arguments | items %}
                {{- arg_name + '="' + arg_val + '"' }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- endif %}
                {%- endfor %}
            {{- ")" }}
        {%- else  %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}
            {{- '{"name": "' + tool_call.name + '", ' }}
            {{- '"parameters": ' }}
            {{- tool_call.arguments | tojson }}
            {{- "}" }}
        {%- endif %}
        {%- if builtin_tools is defined %}
            {{- "<|eom_id|>" }}
        {%- else %}
            {{- "<|eot_id|>" }}
        {%- endif %}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}
{%- endif %}
"""

QWEN_NO_SYSTEM_PROMPT_CHAT_TEMPLATE = """{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- endif %}
    {{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\\n<tool_call>\\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- message.content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
{%- endif %}
"""


def build_row(source: str, prompt: str, response: str, ability: str) -> dict:
    return {
        "data_source": source,
        "ability": ability,
        "reward_model": {
            "style": "rule",
            "ground_truth": "...",
        },
        "extra_info": {
            "prompt": prompt,
            "answer": response,
            "llama_prompt": LLAMA_PROMPT.format(prompt=prompt),
            "qwen_prompt": QWEN_PROMPT.format(prompt=prompt),
        },
    }


def token_count(tokenizer, prompt: str, response: str) -> int:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))


def maybe_add_row(rows: list[dict], qwen_tokenizer, llama_tokenizer, source: str, prompt: str, response: str, args) -> None:
    qwen_len = token_count(qwen_tokenizer, prompt, response)
    llama_len = token_count(llama_tokenizer, prompt, response)
    if qwen_len <= args.max_length and llama_len <= args.max_length:
        rows.append(build_row(source, prompt, response, args.ability))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--magpie_dataset", type=str, default="Magpie-Align/Magpie-Pro-300K-Filtered")
    parser.add_argument("--evol_instruct_dataset", type=str, default="WizardLMTeam/WizardLM_evol_instruct_70k")
    parser.add_argument("--qwen_tokenizer", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--llama_tokenizer", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--magpie_subsample_size", type=int, default=70000)
    parser.add_argument("--magpie_seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="data/instruction_tuning")
    parser.add_argument("--train_file_name", type=str, default="train.parquet")
    parser.add_argument("--val_file_name", type=str, default="val.parquet")
    parser.add_argument("--validation_size", type=int, default=256)
    parser.add_argument("--validation_seed", type=int, default=42)
    parser.add_argument("--shuffle_seed", type=int, default=None)
    parser.add_argument("--ability", type=str, default="coder")
    args = parser.parse_args()

    llama_tokenizer = AutoTokenizer.from_pretrained(args.llama_tokenizer)
    qwen_tokenizer = AutoTokenizer.from_pretrained(args.qwen_tokenizer)
    llama_tokenizer.chat_template = LLAMA_NO_SYSTEM_PROMPT_CHAT_TEMPLATE
    qwen_tokenizer.chat_template = QWEN_NO_SYSTEM_PROMPT_CHAT_TEMPLATE

    rows = []

    magpie = load_dataset(args.magpie_dataset, split="train")
    magpie = magpie.shuffle(seed=args.magpie_seed).select(range(args.magpie_subsample_size))
    for item in tqdm(magpie, desc="Processing data Magpie-Pro-300K-Filtered"):
        conversation = item["conversations"]
        maybe_add_row(
            rows,
            qwen_tokenizer,
            llama_tokenizer,
            "Magpie-Pro-300K-Filtered",
            conversation[0]["value"],
            conversation[1]["value"],
            args,
        )

    evol_instruct = load_dataset(args.evol_instruct_dataset, split="train")
    for item in tqdm(evol_instruct, desc="Processing data WizardLM_evol_instruct_70k"):
        maybe_add_row(
            rows,
            qwen_tokenizer,
            llama_tokenizer,
            "WizardLM_evol_instruct_70k",
            item["instruction"],
            item["output"],
            args,
        )

    df = pd.DataFrame(rows).sample(frac=1, random_state=args.shuffle_seed).reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_dir / args.train_file_name)

    val_size = min(args.validation_size, len(df))
    val_df = df.sample(n=val_size, random_state=args.validation_seed)
    val_df.to_parquet(output_dir / args.val_file_name, index=False)

    print(f"Saved {len(df)} training examples to {output_dir / args.train_file_name}")
    print(f"Saved {val_size} validation examples to {output_dir / args.val_file_name}")


if __name__ == "__main__":
    main()
