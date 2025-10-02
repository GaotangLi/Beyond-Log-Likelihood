"""
Preprocess the Numia dataset to parquet format
"""

import os
import datasets

import argparse


from transformers import AutoTokenizer



LLAMA_PROMPT = '<|begin_of_text|><|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
LLAMA_THINKING_RESPONSE = "<|start_header_id|>think<|end_header_id|>\n\n{thinking}\n\n"
LLAMA_RESPONSE = "<|start_header_id|>answer<|end_header_id|>\n\n{answer}"

QWEN_PROMPT = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n'
QWEN_THINKING_RESPONSE = "<|im_start|>think\n{thinking}\n" 
QWEN_RESPONSE = "<|im_start|>answer\n{answer}"


DEEPSEEK_PROMPT = "<｜begin▁of▁sentence｜>User: {prompt}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\nAssistant:\n"
DEEPSEEK_THINKING_RESPONSE = "thinking:\n {thinking}\n\n"
DEEPSEEK_RESPONSE = "answer:\n {answer}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/medical_sft')
    parser.add_argument('--train_start', type=int, default=0)
    parser.add_argument('--train_end', type=int, default=1000000000)

    args = parser.parse_args()

    data_source = 'UCSC-VLAA/m23k-tokenized'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset['train']
    args.train_end = min(args.train_end, len(train_dataset))
    if args.train_end > 0:
        train_dataset = train_dataset.select(range(args.train_start, args.train_end))

    instruction_following = "\nPlease reason step by step, and put your final answer within \\boxed{{}}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop('prompt')

            question = question_raw + instruction_following

            answer_letter_raw = example.pop('answer_letter')
            answer_string_raw = example.pop('answer_string')  

            answer_thinking_raw = example.pop('reasoning')
            answer_raw = example.pop('distilled_answer_string')

            llama_prompt = LLAMA_PROMPT.format(prompt=question_raw)
            llama_thinking_response = LLAMA_THINKING_RESPONSE.format(thinking=answer_thinking_raw)
            llama_response = LLAMA_RESPONSE.format(answer=answer_raw)

            llama_response_with_thinking = llama_thinking_response + llama_response

            llama_reponse_no_thinking = answer_raw

            qwen_prompt = QWEN_PROMPT.format(prompt=question_raw)
            qwen_thinking_response = QWEN_THINKING_RESPONSE.format(thinking=answer_thinking_raw)
            qwen_response = QWEN_RESPONSE.format(answer=answer_raw)

            qwen_response_with_thinking = qwen_thinking_response + qwen_response
            qwen_reponse_no_thinking = answer_raw

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer_letter_raw},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer_letter": answer_letter_raw,
                    "answer_string": answer_string_raw,
                    "answer_thinking": answer_thinking_raw,
                    "answer_raw": answer_raw,
                    "question": question_raw,
                    "question_instruction": question,
                    "qwen_prompt": qwen_prompt,
                    "qwen_response_with_thinking": qwen_response_with_thinking,
                    "llama_prompt": llama_prompt,
                    "llama_response_with_thinking": llama_response_with_thinking,
                    "llama_reponse_no_thinking": llama_reponse_no_thinking,
                    "qwen_reponse_no_thinking": qwen_reponse_no_thinking,
                },
            }
            return data

        return process_fn
    

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    
    print(f"length of train_dataset: {len(train_dataset)}")
    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    print(train_dataset[0])
