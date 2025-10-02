"""
Preprocess the Numia dataset to parquet format
"""

import os
import datasets

import argparse

from tqdm import tqdm
import reasoning_gym
from transformers import AutoTokenizer

from datasets import Dataset, DatasetDict  # Add DatasetDict import


LLAMA_PROMPT = '<|begin_of_text|><|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an expert at decoding Figlet Font ASCII art. When given Figlet Font text, identify the word it represents and output only that without any explanations or additional text.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

QWEN_PROMPT = '<|im_start|>system\nYou are an expert at decoding Figlet Font ASCII art. When given Figlet Font text, identify the word it represents and output only that without any explanations or additional text.<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/figlet_font')
    parser.add_argument('--train_start', type=int, default=0)
    parser.add_argument('--train_end', type=int, default=1000000000)

    args = parser.parse_args()

    all_data_train = [] 
    dataset = reasoning_gym.create_dataset('figlet_font', size=10, seed=42,  min_word_len=2, max_word_len=5, space_letters=True)

    for i in tqdm(range(40000), desc="Processing train data"):
        question = dataset[i]['question']
        answer = dataset[i]['answer']
        metadata = dataset[i]['metadata']

        curr_data = {
            "ability": "figlet_font", 
            "prompt": [
                {
                    "role": "user",
                    "content": question
                }
            ],
            "answer": answer,
            "qwen_prompt": QWEN_PROMPT.format(prompt=question),
            "llama_prompt": LLAMA_PROMPT.format(prompt=question),
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "train",
                "index": i,
                "question": question,
                "qwen_prompt": QWEN_PROMPT.format(prompt=question),
                "llama_prompt": LLAMA_PROMPT.format(prompt=question),
                "answer": answer,
            }
        }

        all_data_train.append(curr_data) 
    
    all_data_test = []
    for i in tqdm(range(40000, 45000), desc="Processing test data"):
        question = dataset[i]['question']
        answer = dataset[i]['answer']
        metadata = dataset[i]['metadata']

        curr_data = {
            "ability": "figlet_font", 
            "prompt": [
                {
                    "role": "user",
                    "content": question
                }
            ],
            "answer": answer,
            "qwen_prompt": QWEN_PROMPT.format(prompt=question),
            "llama_prompt": LLAMA_PROMPT.format(prompt=question),
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "test",
                "index": i,
                "question": question,
                "qwen_prompt": QWEN_PROMPT.format(prompt=question),
                "llama_prompt": LLAMA_PROMPT.format(prompt=question),
                "answer": answer,
            }
        }

        all_data_test.append(curr_data) 
        
    # Create individual datasets
    ds_train = Dataset.from_list(all_data_train)
    ds_test = Dataset.from_list(all_data_test)

    # Combine into a DatasetDict with train and test splits
    dataset_dict = DatasetDict({
        'train': ds_train,
        'test': ds_test
    })

    dataset_dict.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    print(dataset_dict[0])