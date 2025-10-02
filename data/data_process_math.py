"""
Preprocess the Numia dataset to parquet format
"""

import os
import datasets

import argparse


def remove_boxed(s):
    if s is None:
        return None
    if "\\boxed " in s:
        left = "\\boxed "
        if s.startswith(left):
            return s[len(left) :]
        else:
            return None

    left = "\\boxed{"

    if s.startswith(left) and s.endswith("}"):
        return s[len(left) : -1]
    else:
        return None


def last_boxed_only_string(string):
    if string is None:
        return None
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval

def extract_solution(solution_str):
    try:
        boxed_string = last_boxed_only_string(solution_str)
        if boxed_string is None:
            return None
        return remove_boxed(boxed_string)
    except Exception:
        return None



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/numina_cot_PRM')
    parser.add_argument('--train_file', default='your_path')
    parser.add_argument('--train_start', type=int, default=0)
    parser.add_argument('--train_end', type=int, default=67000)

    args = parser.parse_args()

    dataset = datasets.load_dataset("parquet", data_files=args.train_file)
    train_dataset = dataset['train'] 
    args.train_end = min(args.train_end, len(train_dataset))

    if args.train_end > 0:
        train_dataset = train_dataset.select(range(args.train_start, args.train_end))

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop('question')

            question = question_raw + ' ' + instruction_following

            answer_raw = example.pop('answer')
            solution = extract_solution(answer_raw)

            data = {
                "data_source": "AI-MO/NuminaMath-CoT-processed",
                "prompt": [
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn
    

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    
    print(f"length of train_dataset: {len(train_dataset)}")
    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    print(train_dataset[0])
