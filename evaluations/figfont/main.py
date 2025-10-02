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




def jaro_winkler_similarity(s1, s2, normalize=True, prefix_weight=0.1):
    """
    Jaro-Winkler similarity - particularly good for short strings.
    """
    if normalize:
        s1, s2 = s1.lower().strip(), s2.lower().strip()
    
    if s1 == s2:
        return 1.0
    
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    
    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0
    
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    
    matches = 0
    transpositions = 0
    
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break
    
    if matches == 0:
        return 0.0
    
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    
    jaro = (matches/len1 + matches/len2 + (matches - transpositions/2)/matches) / 3
    
    prefix_len = 0
    for i in range(min(len1, len2)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break
    prefix_len = min(4, prefix_len)
    
    jw_similarity = jaro + prefix_len * prefix_weight * (1 - jaro)
    
    return jw_similarity


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
    curr_jaro_winkler_similarity = 0 

    for i in range(len(input_data)): 
        curr_data = input_data.iloc[i]
        answer = curr_data["extra_info"]["answer"]
        output = outputs[i]
        curr_item = {
            "question": curr_data["extra_info"]["question"],
            "answer": answer,
            "output": output,
            "score": output == answer,
            "jaro_winkler_similarity": jaro_winkler_similarity(output, answer),
            "normalized_score": output.lower() == answer.lower()
        }
        curr_normalized_score += curr_item['normalized_score']
        total_count += 1
        curr_score += curr_item['score']
        curr_jaro_winkler_similarity += curr_item['jaro_winkler_similarity']
        output_data.append(curr_item)

    print(f"Total score: {curr_score / total_count}")
    print(f"Total normalized score: {curr_normalized_score / total_count}")
    print(f"Total jaro_winkler_similarity: {curr_jaro_winkler_similarity / total_count}")
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
    score_json['average_jaro_winkler_similarity'] = curr_jaro_winkler_similarity / total_count
    json.dump(score_json, open(output_path.parent / f"{args.output_file_name}_score.json", 'w'), indent=4)

    print(f"Average score: {score_json['average_score']:.4f}")
    print(f"Average normalized score: {score_json['average_normalized_score']:.4f}")
    print(f"Average jaro_winkler_similarity: {score_json['average_jaro_winkler_similarity']:.4f}")

if __name__ == "__main__":
    main()