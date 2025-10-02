import argparse
import os
import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd

def load_file(input_fp):
    with open(input_fp, 'r') as f:
        data = json.load(f)
    input_data = []
    if isinstance(data, list):
        data = {'normal': data}
    for k,v in data.items():
        for da in v:
            da['source'] = k
        input_data.extend(v)
    return input_data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default="Qwen/Qwen2.5-7B")
    parser.add_argument('--eval_file', type=str, default='evaluations/figfont/data/test.parquet')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--tensor_parallel_size', type=int, default=1) 
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95)  
    parser.add_argument('--max_tokens', type=int, default=5)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument("--model_save_name", type=str, required=True)
    parser.add_argument('--output_file_name', type=str, default='raw_results')
    args = parser.parse_args()
    return args

def get_model(args):
    model = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


LLAMA_PROMPT = '<|begin_of_text|><|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an expert at decoding Figlet Font ASCII art. When given Figlet Font text, identify the word it represents and output only that without any explanations or additional text.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

QWEN_PROMPT = '<|im_start|>system\nYou are an expert at decoding Figlet Font ASCII art. When given Figlet Font text, identify the word it represents and output only that without any explanations or additional text.<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n'


def main():

    args = get_args()

    model, tokenizer = get_model(args)

    input_data = pd.read_parquet(args.eval_file)
    input_prompt = input_data["extra_info"].apply(lambda x: x["question"]).tolist()
    

    
    if args.debug:
        input_prompt = input_prompt[:10]

    if "llama" in args.model_name.lower():
        input_prompt = [LLAMA_PROMPT.format(prompt=item) for item in input_prompt]
    elif "qwen" in args.model_name.lower():
        input_prompt = [QWEN_PROMPT.format(prompt=item) for item in input_prompt]
    else:
        raise NotImplementedError(f"Model {args.model_name} not supported")

    
    sampling_params = SamplingParams(
        temperature=args.temperature, 
        max_tokens=args.max_tokens
    )
    outputs = model.generate(input_prompt, sampling_params)
    outputs = [output.outputs[0].text for output in outputs]

    output_data = []
    curr_score = 0
    curr_normalized_score = 0
    total_count = 0

    for i in range(len(input_data)): 
        curr_data = input_data.iloc[i]
        answer = curr_data["extra_info"]["answer"]
        output = outputs[i]
        curr_item = {
            "question": curr_data["extra_info"]["question"],
            "answer": answer,
            "output": output,
            "score": output == answer,
            "normalized_score": output.lower() == answer.lower()
        }
        curr_normalized_score += curr_item['normalized_score']
        total_count += 1
        curr_score += curr_item['score']
        output_data.append(curr_item)

    print(f"Total score: {curr_score / total_count}")
    print(f"Total normalized score: {curr_normalized_score / total_count}")
    print(f"Total count: {total_count}, total length: {len(input_data)}, total score: {curr_score}")

    if args.debug:
        for output in outputs:
            print(output)
            print("-"*100)
            print("\n")
        exit()


    output_path = Path("./results") / "figfont" / args.model_save_name / f"{args.output_file_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(output_data, open(output_path, 'w'), indent=4)


    score_json = {}
 
    score_json['average_score'] = curr_score / total_count
    score_json['average_normalized_score'] = curr_normalized_score / total_count
    json.dump(score_json, open(output_path.parent / f"{args.output_file_name}_score.json", 'w'), indent=4)

    print(f"Average score: {score_json['average_score']:.4f}")
    print(f"Average normalized score: {score_json['average_normalized_score']:.4f}")


if __name__ == "__main__":
    main()